import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16
from tdt import read_block
from tqdm import tqdm

from guppy.extractors import TdtRecordingExtractor


def read_tsq(tsq_file_path):
    names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
    formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)
    tsq = np.fromfile(tsq_file_path, dtype=tsq_dtype)
    df = pd.DataFrame(tsq)
    return df


def stub_tev_file(tev_file_path, df, stubbed_file_path, stream_name_to_num_segments):
    stream_names = list(stream_name_to_num_segments.keys())
    with open(tev_file_path, "r+b") as f:
        content = f.read()
    if os.path.exists(stubbed_file_path):
        os.remove(stubbed_file_path)

    print("Reading file starts and stops")
    all_starts, all_stops, all_stream_names = [], [], []
    for stream_name in stream_names:
        row = df["name"] == stream_name
        allIndexesWhereEventIsPresent = np.where(row == 1)
        first_row = allIndexesWhereEventIsPresent[0][0]
        nsample = df["size"][first_row] - 10
        fp_loc = np.asarray(df["fp_loc"][allIndexesWhereEventIsPresent[0]])
        print(stream_name, len(fp_loc))
        for loc in fp_loc:
            all_starts.append(loc)
            all_stops.append(loc + nsample * 4)
            all_stream_names.append(stream_name)
    sort_idx = np.argsort(all_starts)
    all_starts = np.array(all_starts)[sort_idx]
    all_stops = np.array(all_stops)[sort_idx]
    all_stream_names = np.array(all_stream_names)[sort_idx]

    print("Writing stubbed file")
    previous_stop = 0
    write_position = 0
    original_to_new_fp_loc = {}
    stream_name_to_num_written = {stream_name: 0 for stream_name in stream_names}
    for start, stop, stream_name in zip(all_starts, all_stops, all_stream_names):
        with open(stubbed_file_path, "a+b") as f:
            gap = content[previous_stop:start]
            f.write(gap)
            write_position += len(gap)
            num_written = stream_name_to_num_written[stream_name]
            num_segments = stream_name_to_num_segments[stream_name]
            if num_written < num_segments:
                original_to_new_fp_loc[start] = write_position
                segment = content[start:stop]
                f.write(segment)
                write_position += len(segment)
                stream_name_to_num_written[stream_name] += 1
            previous_stop = stop
    with open(stubbed_file_path, "a+b") as f:
        f.write(content[previous_stop:])
    return original_to_new_fp_loc


def stub_tsq_file(tsq_file_path, stubbed_tsq_file_path, stream_name_to_num_segments, original_to_new_fp_loc):
    stream_names = list(stream_name_to_num_segments.keys())
    names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
    formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)
    tsq_data = np.fromfile(tsq_file_path, dtype=tsq_dtype)

    rows_to_keep_mask = []
    stream_name_to_num_kept_rows = {stream_name: 0 for stream_name in stream_names}
    for row in tqdm(tsq_data):
        stream_name = row["name"]
        if stream_name not in stream_names:
            keep_row = True
            rows_to_keep_mask.append(keep_row)
            continue
        num_kept_rows = stream_name_to_num_kept_rows[stream_name]
        num_segments = stream_name_to_num_segments[stream_name]
        if num_kept_rows < num_segments:
            keep_row = True
            stream_name_to_num_kept_rows[stream_name] += 1
        else:
            keep_row = False
        rows_to_keep_mask.append(keep_row)
    rows_to_keep_mask = np.array(rows_to_keep_mask)
    stubbed_tsq_data = tsq_data[rows_to_keep_mask]

    # correct timestamps
    first_stream_name = stream_names[0]
    stubbed_last_timestamp = stubbed_tsq_data["timestamp"][stubbed_tsq_data["name"] == first_stream_name][-1]
    stubbed_tsq_data["timestamp"][stubbed_tsq_data["name"] == b"\x02"] = stubbed_last_timestamp

    # correct fp_loc: positions shifted because skipped segments were removed from the TEV file
    stream_mask = np.isin(stubbed_tsq_data["name"], list(stream_name_to_num_segments.keys()))
    stubbed_tsq_data["fp_loc"][stream_mask] = [
        original_to_new_fp_loc[loc] for loc in stubbed_tsq_data["fp_loc"][stream_mask]
    ]

    # correct size: first record's size field = total file size in bytes
    record_size = np.dtype(tsq_dtype).itemsize
    stubbed_tsq_data["size"][0] = len(stubbed_tsq_data) * record_size

    stubbed_tsq_data.tofile(stubbed_tsq_file_path)


def main():
    session_folder_path = Path(
        "/Users/pauladkisson/Documents/CatalystNeuro/Guppy/GuPPy/stubbed_testing_data/tdt/Photo_63_207-181030-103332"
    )
    tev_file_path = session_folder_path / "Cohort6_Photo_63_207-181030-103332.tev"
    tsq_file_path = session_folder_path / "Cohort6_Photo_63_207-181030-103332.tsq"

    stubbed_session_folder_path = session_folder_path.parent / "Cohort6_Photo_63_207-181030-103332_stubbed"
    if os.path.exists(stubbed_session_folder_path):
        shutil.rmtree(stubbed_session_folder_path)
    shutil.copytree(session_folder_path, stubbed_session_folder_path)
    os.remove(stubbed_session_folder_path / "Cohort6_Photo_63_207-181030-103332.tev")
    os.remove(stubbed_session_folder_path / "Cohort6_Photo_63_207-181030-103332.tsq")

    df = read_tsq(tsq_file_path)
    stubbed_file_path = stubbed_session_folder_path / "Cohort6_Photo_63_207-181030-103332_stubbed.tev"
    stubbed_tsq_file_path = stubbed_session_folder_path / "Cohort6_Photo_63_207-181030-103332_stubbed.tsq"
    stream_name_to_num_segments = {
        b"Dv1A": 10,
        b"Dv2A": 10,
        b"Dv3B": 10,
        b"Dv4B": 10,
        b"Fi1i": 60,
        b"Fi1r": 60,
        b"LNRW": 10,
        b"LNnR": 10,
        b"PrtN": 10,
        b"PrtR": 10,
        b"RNPS": 10,
    }
    original_to_new_fp_loc = stub_tev_file(tev_file_path, df, stubbed_file_path, stream_name_to_num_segments)
    stub_tsq_file(tsq_file_path, stubbed_tsq_file_path, stream_name_to_num_segments, original_to_new_fp_loc)

    # Check with tdt read_block
    tdt_data = read_block(session_folder_path, t2=1).streams["Dv1A"].data
    stubbed_tdt_photometry = read_block(stubbed_session_folder_path, t2=1)
    stubbed_tdt_data = stubbed_tdt_photometry.streams["Dv1A"].data
    np.testing.assert_equal(stubbed_tdt_data, tdt_data[: len(stubbed_tdt_data)])

    # Check with TdtRecordingExtractor
    recording_extractor = TdtRecordingExtractor(folder_path=session_folder_path)
    output_dicts = recording_extractor.read(events=["Dv1A"], outputPath="")
    S = output_dicts[0]
    raw_data = S["data"]
    stubbed_recording_extractor = TdtRecordingExtractor(folder_path=stubbed_session_folder_path)
    stubbed_output_dicts = stubbed_recording_extractor.read(events=["Dv1A"], outputPath="")
    stubbed_S = stubbed_output_dicts[0]
    stubbed_data = stubbed_S["data"]
    np.testing.assert_equal(stubbed_data, raw_data[: len(stubbed_data)])


if __name__ == "__main__":
    main()
