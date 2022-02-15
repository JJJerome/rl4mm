from invoke import task


MODULES_TO_CHECK = ["RL4MM", "*.py"]
MODULES_TO_CHECK_STR = " ".join(MODULES_TO_CHECK)
BLACK_PATHS_TO_IGNORE = []
BLACK_PATHS_TO_IGNORE_STR = " ".join(BLACK_PATHS_TO_IGNORE)
MYPY_PATHS_TO_IGNORE = ["RL4MM/simulator/", "RL4MM/features/"]
MYPY_EXCLUSION_STR = ""
for path in MYPY_PATHS_TO_IGNORE:
    MYPY_EXCLUSION_STR += " --exclude " + path


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
    print("Running flake8...")
    c.run(f"flake8 {MODULES_TO_CHECK_STR}")
    print("No flake8 errors")
    print("Running mypy...")
    c.run(f"mypy -p {MODULES_TO_CHECK[0]}" + MYPY_EXCLUSION_STR)
    print("No mypy errors")
    c.run("python check_init_files.py")
    c.run("python check_all_py_imports.py")
