#segemnet by time table
import numpy as np
import pandas as pd
from datetime import timedelta,datetime
import re

FS = 125
TIME_OFFSET_SEC = 19
offset_points = TIME_OFFSET_SEC * FS

path = r"data/train_data/raw"
time_table_name = r"data/train_data/fn_rx_timetable.xlsx"
demo_data_name = r"20240805T193613.csv"
OUT_DIR = r"data/train_data/segment_datas"

#list
def dateTime2index(dateTime_arr,timePoints,label):
    index = []
    if label == 'st':
        for time in dateTime_arr:
            index.append(np.where(timePoints>=time)[0][0])
    if label == 'et':
        for time in dateTime_arr:
            if np.where(timePoints>time)[0].size>0:
                index.append(np.where(timePoints>time)[0][0])
            else:
                index.append(len(timePoints)-1)
    return index

def seg_data(FS,TIME_OFFSET_SEC,path,demo_data_name,time_table_name):
    offset_points = TIME_OFFSET_SEC * FS
    demo_data_path = path+'/'+demo_data_name
    demo_data_name = re.search(r'(\d{8})T(\d{6})',demo_data_name).group()
    demo_data = np.array(pd.read_csv(demo_data_path))
    time_table = pd.read_excel(time_table_name,sheet_name=None,index_col=0,header=None)
    t = np.array(demo_data)[:,0]
    demo_eeg = np.array(demo_data)[:,1:]

    start_time = datetime.strptime(demo_data_name,"%Y%m%dT%H%M%S")
    start_time_np = np.datetime64(start_time,'ms')
    timePoints = start_time_np+np.array(t*1e3,dtype='timedelta64[ms]')
    
    label_col = time_table['Sheet1'].iloc[:, 1].astype(str).str.strip()
    #find the index of two classes
    index_fn = np.flatnonzero(label_col.eq("find_number").to_numpy())
    index_rx = np.flatnonzero(label_col.eq("relax").to_numpy() | label_col.str.lower().eq("relax").to_numpy())

    if len(index_fn) <= len(index_rx):
        count = len(index_fn)
    else:
        count = len(index_rx)

    for trail_id in range(count):
        fn_time_range = time_table['Sheet1'].iloc[index_fn[trail_id],0].replace("：", ":")
        rx_time_range = time_table['Sheet1'].iloc[index_rx[trail_id],0].replace("：", ":")
        all_list = np.column_stack((np.array(fn_time_range), np.array(rx_time_range)))
        fn_start,fn_end = parse_time(fn_time_range)
        rx_start,rx_end = parse_time(rx_time_range)
        fn_start_dt = combine_date_time(start_time, fn_start)
        fn_end_dt = combine_date_time(start_time, fn_end)
        rx_start_dt = combine_date_time(start_time, rx_start)
        rx_end_dt = combine_date_time(start_time, rx_end)

        st = dateTime2index([fn_start_dt,rx_start_dt],timePoints,'st')
        et = dateTime2index([fn_end_dt,rx_end_dt],timePoints,'et')

        fn_start = np.array([datetime.strptime(x.split('-')[0],'%H:%M:%S').time() for x in all_list.flatten()])
        rx_end = np.array([datetime.strptime(x.split('-')[1],'%H:%M:%S').time() for x in all_list.flatten()])

        fn_duration_sec = (np.datetime64(fn_end_dt,"ms")-np.datetime64(fn_start_dt,"ms")) / np.timedelta64(1,"s")
        fn_grades = 111-fn_duration_sec//9.8
        fn_eeg = demo_eeg[max(st[0] - offset_points, 0):max(et[0] - offset_points, 0), :].T
        rx_eeg = demo_eeg[max(st[1] - offset_points, 0):max(et[1] - offset_points, 0), :].T

        return fn_eeg,rx_eeg,fn_grades

def parse_time(time_range):
    time_range = str(time_range).strip().replace("：", ":")
    start_str, end_str = time_range.split("-")
    start_t = datetime.strptime(start_str.strip(), "%H:%M:%S").time()
    end_t = datetime.strptime(end_str.strip(), "%H:%M:%S").time()
    return start_t,end_t

def combine_date_time(base_datetime, clock_time):
    return base_datetime + timedelta(
        hours=clock_time.hour,
        minutes=clock_time.minute,
        seconds=clock_time.second,
    )

def main():
    fn_eeg,rx_eeg,fn_grades = seg_data(FS,TIME_OFFSET_SEC,path,demo_data_name,time_table_name)
    np.save(OUT_DIR+'/'+'find_numbers_segment.npy', fn_eeg)
    np.save(OUT_DIR+'/'+'fn_grades.npy', np.array(fn_grades))
    np.save(OUT_DIR+'/'+'relax_numbers_segment.npy', rx_eeg)


if __name__ == "__main__":
    main()