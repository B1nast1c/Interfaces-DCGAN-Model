from google_images_downloader import GoogleImagesDownloader
from utils import common


def download():
    downloader = GoogleImagesDownloader(browser="chrome", show=False, debug=False,
                                        quiet=False, disable_safeui=False)
    for keyword in common.CLASSES:
        downloader.download(
            keyword,
            limit=100,  # Cantidad de prueba para la posterior extracción
            file_format="JPEG",
            destination=common.IMAGES_LOCATION
        )
    downloader.close()
