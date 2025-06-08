from binance_bulk_downloader.downloader import BinanceBulkDownloader

# generate instance
downloader = BinanceBulkDownloader(asset="um", data_type="klines", data_frequency="1h", destination_dir="./crypto/")
downloader.run_download()