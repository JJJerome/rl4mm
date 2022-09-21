from dotenv import find_dotenv, load_dotenv


def setup_environment_variables(dotenv_filename: str = ".env") -> None:
    try:
        dotenv_path = find_dotenv(filename=dotenv_filename, raise_error_if_not_found=True)
    except IOError:
        raise IOError(f"Could not find dotenv file: {dotenv_filename}")
    load_dotenv(dotenv_path=dotenv_path, override=False)
