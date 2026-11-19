"""Checks of the .mat loader on small files whose content is known.  Run:  python -m pytest tests"""
import h5py
import numpy as np
import pytest
import scipy.io as sio

from motility import io

DT = 0.05


def make_tracks():
    """Three tracks: a straight line, one with an interior NaN gap, one with leading and trailing NaNs."""
    t1 = np.arange(10) * DT
    x1, y1 = 1.0 * np.arange(10), np.zeros(10)                       # 20 µm/s along x
    t2 = 2.0 + np.arange(8) * DT
    x2, y2 = np.arange(8, dtype=float), np.arange(8, dtype=float)
    x2[3:5] = np.nan                                                 # gap of 2 frames
    t3 = 5.0 + np.arange(6) * DT
    x3, y3 = np.arange(6, dtype=float), np.zeros(6)
    x3[0] = x3[-1] = np.nan                                          # leading and trailing
    return [(x1, y1, t1), (x2, y2, t2), (x3, y3, t3)]


def write_classic(path, tracks):
    arr = np.array([(x, y, t) for x, y, t in tracks], dtype=[("x", "O"), ("y", "O"), ("time", "O")])
    sio.savemat(path, {"tracks": arr})


def write_v73(path, tracks):
    """Mimic the HDF5 layout of a MATLAB -v7.3 struct array: a group with one dataset of object references
    per field, and the arrays stored transposed (column-major)."""
    with h5py.File(path, "w") as h:
        refs = h.create_group("#refs#")
        grp = h.create_group("tracks")
        for name, j in (("x", 0), ("y", 1), ("time", 2)):
            ds = grp.create_dataset(name, (1, len(tracks)), dtype=h5py.ref_dtype)
            for i, tr in enumerate(tracks):
                d = refs.create_dataset(f"{name}{i}", data=np.asarray(tr[j])[None, :].T)
                ds[0, i] = d.ref


@pytest.mark.parametrize("writer", [write_classic, write_v73])
def test_read_mat_gives_struct_array(tmp_path, writer):
    tracks = make_tracks()
    path = tmp_path / "t.mat"
    writer(path, tracks)
    data = io.read_mat(path)
    assert list(data) == ["tracks"]
    assert len(data["tracks"]) == 3
    assert set(data["tracks"][0]) == {"x", "y", "time"}
    np.testing.assert_allclose(data["tracks"][0]["x"], tracks[0][0])
    np.testing.assert_allclose(data["tracks"][1]["time"], tracks[1][2])


@pytest.mark.parametrize("writer", [write_classic, write_v73])
def test_load_mat_tracks_drop(tmp_path, writer):
    path = tmp_path / "t.mat"
    writer(path, make_tracks())
    df, meta = io.load_mat_tracks(path)
    assert list(df.columns) == ["track_id", "frame", "t", "x_um", "y_um"]
    assert meta["fps"] == pytest.approx(20.0)
    assert meta["n_tracks"] == 3
    assert meta["n_detections"] == 10 + 6 + 4                        # gap dropped, edges cut
    assert meta["n_missing_positions"] == 2 and meta["longest_gap_frames"] == 2
    assert not df[["x_um", "y_um"]].isna().any().any()
    two = df[df.track_id == 1]
    assert list(two["frame"]) == [40, 41, 42, 45, 46, 47]            # frames count the whole movie, gap visible
    three = df[df.track_id == 2]
    assert len(three) == 4 and three["x_um"].iloc[0] == 1.0
    assert meta["track_duration_max_s"] == pytest.approx(9 * DT)


def test_load_mat_tracks_interpolate(tmp_path):
    path = tmp_path / "t.mat"
    write_classic(path, make_tracks())
    df, meta = io.load_mat_tracks(path, missing="interpolate")
    two = df[df.track_id == 1]
    assert list(two["frame"]) == list(range(40, 48))
    np.testing.assert_allclose(two["x_um"], np.arange(8))            # linear gap filled exactly
    assert meta["n_missing_positions"] == 2 and meta["missing"] == "interpolate"


def test_load_mat_tracks_errors(tmp_path):
    path = tmp_path / "t.mat"
    write_classic(path, make_tracks())
    with pytest.raises(KeyError):
        io.load_mat_tracks(path, var="cells")
    with pytest.raises(ValueError):
        io.load_mat_tracks(path, missing="fill")


def test_describe_mat_prints_variables(tmp_path, capsys):
    path = tmp_path / "t.mat"
    write_classic(path, make_tracks())
    io.describe_mat(path)
    out = capsys.readouterr().out
    assert "MATLAB" in out and "tracks: struct array, 3 elements" in out and ".x" in out


def test_mat_files_summary(tmp_path):
    write_classic(tmp_path / "b.mat", make_tracks())
    write_classic(tmp_path / "a.mat", make_tracks()[:1])
    table = io.mat_files_summary(tmp_path)
    assert list(table["file"]) == ["a.mat", "b.mat"]
    assert list(table["n_tracks"]) == [1, 3]
    assert table["fields"].iloc[0] == "x, y, time"
    assert table["fps"].iloc[1] == pytest.approx(20.0)
    assert table["movie_s"].iloc[1] == pytest.approx(5.0 + 5 * DT, abs=0.05)   # rounded to 0.1 s
    assert io.mat_files_summary(tmp_path / "empty").empty
