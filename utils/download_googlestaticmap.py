"""This downloader downloads satellite images from the Google Static Maps API.

Usage:
    $ python download_googlestaticmap.py \
    >   --log LOG_FILE.csv \
    >   --initialize INIT_FILE.csv
    $ nohup python download_googlestaticmap.py \
    >   --log LOG_FILE.csv \
    >   --num 3 \
    >   --download-dir DIR \
    >   > logs/download_googlestaticmap.log &
"""

import os
import pandas as pd
import requests
from argparse import ArgumentParser
from tqdm import tqdm


class Downloader(object):
    """This class keeps a log of the downloading process,
    checks for duplicates and manages bad HTTP requests.

    Args:
        queue (pandas.DataFrame): Log of the downloaded objects.
    """

    def __init__(self, queue=None):
        # if downloading for the first time
        if queue is None:
            # create an empty queue
            self.queue = pd.DataFrame(columns=['index', 'url', 'status'])
            self.queue.set_index('index', inplace=True)
            self.queue.index.name = 'index'
        # if not, load previous log
        else:
            self.queue = queue

    def request(self, indices, mapping):
        """This method requests objects to be downloaded and adds them to the queue.

        Args:
            indices (numpy.array): unique id for each object in the queue.
            mapping (callable): takes in the indices and generates the urls.
        """
        urls = [mapping(index) for index in indices]
        subqueue = pd.DataFrame(
            {'url': urls,
             'status': False},
            index=indices)
        subqueue.index.name = 'index'
        try:
            self.queue = pd.concat([self.queue, subqueue],
                                   verify_integrity=True)
            print('{} new requests initiated.'.format(subqueue.shape[0]))
        except ValueError:
            raise Exception('Overlapping new requests with existing requests.')

    def download(self, num, download_dir,
                 test_page='https://www.google.com',
                 suffix='.png', min_size=20000):
        """This method downloads objects.

        Args:
            num (int): number of downloads to perform.
            download_dir (str): downloading directory.
            test_page (str): url to try in order to check internet connection.
            suffix (str): suffix for saved files.
            min_size (int): minimum file size. Helps drop NA images.
        """

        # check local directory
        if not os.path.isdir(download_dir):
            raise Exception('Download directory does not exist.')

        # check internet connection
        _ = requests.get(test_page, timeout=1)

        # extract items already downloaded
        mask = self.queue['status']
        if not mask.all():
            # number of files to be downloaded
            update_num = min((~mask).sum(), num)
            print('Preparing to download {} files.'.format(update_num))
            idxs = self.queue[~mask].index.copy()
            idxs = idxs[0:update_num]
            # downloading starts
            for idx in tqdm(idxs):
                # fetch url
                url = self.queue.loc[idx, 'url']
                # construct file names
                file_name = os.path.join(download_dir, ''.join([idx, suffix]))
                # check if file exists already
                if os.path.isfile(file_name):
                    # update status
                    self.queue.loc[idx, 'status'] = True
                    print('{} already exists.'.format(file_name))
                else:
                    r = requests.get(url)
                    if int(r.headers['Content-Length']) > min_size:
                        with open(file_name, 'wb') as f:
                            _ = f.write(r.content)
                        # update status
                        self.queue.loc[idx, 'status'] = True
                        print('{} successfully downloaded.'.format(file_name))
                    else:
                        print('{} skipped - file too small: {} bytes.'.format(
                            file_name, int(r.headers['Content-Length'])))
                        self.queue.drop(idx, inplace=True)
        if mask.all():
            print('Downloading completed.')


def make_url(idx):
    """Helper function to generate the urls for the Google Static Maps API.

    Args:
        index (str): Identifies an image.

    Returns:
        url (str): The URL to the image.
    """
    params = {
        'center': ('{:.6f},{:.6f}'
                   .format(df.loc[idx, 'lat'], df.loc[idx, 'lon'])),
        'zoom': '19',
        'size': '640x640',
        'scale': '2',
        'maptype': 'satellite',
        'key': GOOGLE_API_KEY}
    params_str = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])
    return '?'.join(['https://maps.googleapis.com/maps/api/staticmap',
                     params_str])


if __name__ == '__main__':
    # parse arguments passed from the command line
    parser = ArgumentParser(
        description='Downloads satellite images from Google Statics Maps API.')
    parser.add_argument('--log', default=None, type=str,
                        help='name of log file (.csv)')
    # request
    parser.add_argument('--initialize', default=None, type=str,
                        help='a new list of files to be downloaded')
    parser.add_argument(
        '--api-key', default='GOOGLE_API_KEY.txt',
        help='file that stores the API key, defaults to GOOGLE_API_KEY.txt')
    # download
    parser.add_argument(
        '--num', default=None, type=int,
        help='number of downloads to perform, this flag turns on downloading')
    parser.add_argument('--download-dir', default=None, type=str,
                        help='downloading directory')
    # parse
    args = parser.parse_args()

    assert args.log is not None, 'Input log file path!'

    # parse and make url list
    if args.initialize is not None:
        d = Downloader()
        # fetch authentication key
        with open(args.api_key, 'r') as f:
            GOOGLE_API_KEY = f.read()
        # read coordinates and index
        df = pd.read_csv(args.initialize, index_col='index')
        df = df.filter(items=['lon', 'lat'])
        d.request(indices=df.index.values, mapping=make_url)
    else:
        q = pd.read_csv(args.log, index_col='index')
        d = Downloader(queue=q)

    # download
    if args.num is not None:
        assert args.download_dir is not None, 'Input download directory!'
        d.download(num=args.num, download_dir=args.download_dir)

    # save the log
    d.queue.to_csv(args.log)
