import pandas as pd
def analyze_venue_categories(file_path):
    # 读取TSV文件，使用与data_preparation.py相同的参数
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=[
            'user_id', 'venue_id', 'venue_category_id', 'venue_category_name',
            'latitude', 'longitude', 'timezone_offset', 'utc_time'
        ],
        encoding='ISO-8859-1'
    )

    # 统计不同关键词数量（直接统计第四列的唯一值）
    distinct_count = df['venue_category_name'].nunique()

    # 统计总关键词出现次数（即数据集行数）
    total_count = len(df)

    print(f"Distinct venue categories: {distinct_count}")
    print(f"Total venue category records: {total_count}")


if __name__ == "__main__":
    analyze_venue_categories("dataset/dataset_TSMC2014_TKY.txt")