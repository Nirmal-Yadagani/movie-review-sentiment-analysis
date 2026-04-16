import sys
import traceback

def error_message_detail(error, error_detail):
  """
  Extracts file name and line number from the traceback.
  """
  _, _, exc_tab = error_detail.exc_info()

  # Extract details safely
  file_name = "Unknown"
  line_number = 0

  if exc_tab is not None:
    tb_list = traceback.extract_tb(exc_tab)
    last_traceback = tb_list[-1]
    file_name = last_traceback.filename
    line_number = last_traceback.lineno

  raw_message = str(error)

  return f"Error occurred in script: [{file_name}] at line number: [{line_number}] message: [{raw_message}]"


class MyException(Exception):
  def __init__(self, error_message, error_detail: sys):
    """
    :param error_message: error message in string format or Exception object
    :param error_detail: the sys module
    """
    # 1. Initialize the base Exception class first
    super().__init__(str(error_message))

    # 2. Generate the detailed message
    self.error_message = error_message_detail(error_message, error_detail)

  def __str__(self):
    return self.error_message