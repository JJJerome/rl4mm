from invoke import task


MODULES_TO_CHECK = ["RL4MM", "*.py"]
MODULES_TO_CHECK_STR = " ".join(MODULES_TO_CHECK)
BLACK_PATHS_TO_IGNORE = []
BLACK_PATHS_TO_IGNORE_STR = " ".join(BLACK_PATHS_TO_IGNORE)


@task
def black_reformat(c):
    if len(BLACK_PATHS_TO_IGNORE) > 0:
        c.run(f"black --line-length 120 {MODULES_TO_CHECK_STR} --exclude {BLACK_PATHS_TO_IGNORE_STR}")
    else:
        c.run(f"black --line-length 120 {MODULES_TO_CHECK_STR}")


@task
def check_python(c):
    if len(BLACK_PATHS_TO_IGNORE) > 0:
        c.run(f"black --check --line-length 120 {MODULES_TO_CHECK_STR} --exclude {BLACK_PATHS_TO_IGNORE_STR}")
    else:
        c.run(f"black --check --line-length 120 {MODULES_TO_CHECK_STR}")
    c.run(f"flake8 {MODULES_TO_CHECK_STR}")
    c.run(f"mypy {MODULES_TO_CHECK_STR}")
    c.run("python check_init_files.py")
    c.run("python check_all_py_imports.py")
