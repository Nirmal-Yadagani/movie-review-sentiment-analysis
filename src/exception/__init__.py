import sys
import traceback

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts the file name, line number, and error description from a traceback.
    
    Args:
        error (Exception): The raw exception caught during execution.
        error_detail (sys): The sys module, used to extract the current exception info.
        
    Returns:
        str: A highly detailed, formatted string containing the script name, 
             line number, and the exact error message.
    """
    _, _, exc_tab = error_detail.exc_info()

    # Extract details safely with fallbacks
    file_name = "Unknown"
    line_number = 0

    if exc_tab is not None:
        tb_list = traceback.extract_tb(exc_tab)
        last_traceback = tb_list[-1]
        file_name = last_traceback.filename
        line_number = last_traceback.lineno

    raw_message = str(error)

    return f"Error occurred in script: [{file_name}] at line number: [{line_number}] message: [{raw_message}]"


class CustomException(Exception):
    """
    Custom exception class designed to capture detailed traceback information 
    for debugging machine learning pipelines.
    """
    
    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Initializes the CustomException and processes the error details.
        
        Args:
            error_message (Exception): The original error raised.
            error_detail (sys): The sys module to track the traceback details.
        """
        # Initialize the base Exception class first
        super().__init__(str(error_message))

        # Generate and store the detailed error message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the detailed error message when the exception is printed.
        """
        return self.error_message