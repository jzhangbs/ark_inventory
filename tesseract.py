import ctypes
import ctypes.util
import cv2


class TesseractError(Exception):
    pass


class Tesseract(object):
    _lib = None
    _api = None

    class TessBaseAPI(ctypes._Pointer):
        _type_ = type('_TessBaseAPI', (ctypes.Structure,), {})

    @classmethod
    def setup_lib(cls, lib_path=None):
        if cls._lib is not None:
            return
        if lib_path is None:
            lib_path = ctypes.util.find_library('libtesseract-5')
            print(lib_path)
        cls._lib = lib = ctypes.CDLL(lib_path)

        # source:
        # https://github.com/tesseract-ocr/tesseract/blob/95ea778745edd1cdf6ee22f9fe653b9e061d5708/src/api/capi.h

        lib.TessBaseAPICreate.restype = cls.TessBaseAPI

        lib.TessBaseAPIDelete.restype = None # void
        lib.TessBaseAPIDelete.argtypes = (
            cls.TessBaseAPI,) # handle

        lib.TessBaseAPIInit3.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p)

        lib.TessBaseAPISetImage.restype = None
        lib.TessBaseAPISetImage.argtypes = (cls.TessBaseAPI, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

        lib.TessBaseAPISetVariable.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p)

        lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_char_p
        lib.TessBaseAPIGetUTF8Text.argtypes = (
            cls.TessBaseAPI,)

    def __init__(self, language=b'eng', datapath=None, lib_path=None):
        if self._lib is None:
            self.setup_lib(lib_path)
        self._api = self._lib.TessBaseAPICreate()
        # print("initializing tesseract!!!!")
        if self._lib.TessBaseAPIInit3(self._api, datapath, language):
            # print("Tesseract initialization failed!!")
            raise TesseractError('initialization failed')

    def __del__(self):
        if not self._lib or not self._api:
            return
        if not getattr(self, 'closed', False):
            self._lib.TessBaseAPIDelete(self._api)
            self.closed = True

    def _check_setup(self):
        if not self._lib:
            raise TesseractError('lib not configured')
        if not self._api:
            raise TesseractError('api not created')

    def set_image(self, imagedata, width, height,
                  bytes_per_pixel, bytes_per_line=None):
        self._check_setup()
        if bytes_per_line is None:
            bytes_per_line = width * bytes_per_pixel
        # print("bytes per line={}".format(bytes_per_line))
        self._lib.TessBaseAPISetImage(self._api,
                                      imagedata, width, height,
                                      bytes_per_pixel, bytes_per_line)

    def set_variable(self, key, val):
        self._check_setup()
        self._lib.TessBaseAPISetVariable(self._api, key, val)

    def get_utf8_text(self):
        self._check_setup()
        return self._lib.TessBaseAPIGetUTF8Text(self._api)

    def get_text(self):
        self._check_setup()
        result = self._lib.TessBaseAPIGetUTF8Text(self._api)
        if result:
            return result.decode('utf-8')


def convert_to_grayscale(image_data):
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)


# a method to make it look similar to tesslib.py
def tesseract_process_image2(tess, frame_piece):
    grayscaled = len(frame_piece.frame.shape) == 2
    if not grayscaled:
        image_data = convert_to_grayscale(frame_piece.frame)

    height, width = frame_piece.frame.shape
    tess.set_variable(b"debug_file", b'/dev/null')
    tess.set_variable(b"tessedit_pageseg_mode", bytes(frame_piece.psm))
    tess.set_variable(b"tessedit_ocr_engine_mode", bytes(frame_piece.oem))
    if frame_piece.whitelist is not None:
        tess.set_variable(b"tessedit_char_whitelist", frame_piece.whitelist)
    # tess.set_variable(b"user_words_suffix", "user-data")
    # tess.set_variable(b"user_pattern_suffix", "user-pattern")
    # tess.set_variable(b"image_default_resolution", b"70")
    tess.set_image(frame_piece.frame.ctypes, width, height, 1)
    text = tess.get_utf8_text()
    return text.strip()


class FramePiece(object):
  def __init__(self, img, whitelist=None, psm=3, oem=3):
    self.frame = img
    self.whitelist = whitelist
    self.psm = psm
    self.oem = oem


# overloaded method for view page
def tesseract_process_image(tess, frame, whitelist=None):
    frame_piece = FramePiece(frame, whitelist)
    return tesseract_process_image2(tess, frame_piece)

