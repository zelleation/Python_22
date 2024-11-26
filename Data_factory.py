from Data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar
from mindspore.dataset import GeneratorDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    # class mindspore.dataset.GeneratorDataset(
    # source, column_names=None, column_types=None, schema=None,
    # num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None,
    # num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=None)
    data_loader = GeneratorDataset(
        data_set,
        column_names = ["data_x", "data_y", "data_stamp"],

        shuffle=shuffle_flag,
        num_parallel_workers=args.num_workers,
        )
    data_loader.batch(batch_size,drop_remainder = drop_last)

    return data_set, data_loader
