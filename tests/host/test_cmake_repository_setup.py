# Copyright (c) 2026 liyinlong. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the real CMake installer with isolated package-manager commands."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


INSTALLER = Path(__file__).resolve().parents[2] / "install_deps.sh"
KEY = b"-----BEGIN PGP PUBLIC KEY BLOCK-----\nfixture\n-----END PGP PUBLIC KEY BLOCK-----\n"
KEYRING = b"fixture-binary-keyring"
REPOSITORY = (
    b"deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] "
    b"https://apt.kitware.com/ubuntu/ bionic main\n"
)
COMMAND = r"""import json, os, sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["CMAKE_SETUP_FIXTURE"])
failure = os.environ.get("CMAKE_SETUP_FAILURE", "")
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps([name, *args]) + "\n")

if name == "cmake":
    installed = (root / "installed").exists() or failure == "already_installed"
    sys.stdout.write("cmake version " + ("3.27.0" if installed else "3.10.2") + "\n")
elif name == "grep":
    sys.exit(1 if failure == "other_debian" else 0)
elif name == "wget":
    if failure == "download":
        sys.exit(7)
    sys.stdout.buffer.write((root / "public-key").read_bytes())
elif name == "gpg":
    key = sys.stdin.buffer.read()
    (root / "gpg-input").write_bytes(key)
    if key != (root / "public-key").read_bytes():
        sys.stderr.write("invalid public key input\n")
        sys.exit(3)
    if failure == "dearmor":
        sys.exit(11)
    sys.stdout.buffer.write(b"fixture-binary-keyring")
elif name == "sudo":
    if args[0] == "tee":
        data = sys.stdin.buffer.read()
        output = "keyring" if args[1].endswith(".gpg") else "repository"
        if failure == output:
            sys.exit(13 if output == "keyring" else 17)
        (root / output).write_bytes(data)
        sys.stdout.buffer.write(data)
    elif args[:2] == ["apt", "update"]:
        if failure == "update":
            sys.exit(19)
    elif args[:2] == ["apt", "install"]:
        if failure == "install":
            sys.exit(23)
        (root / "installed").touch()
    else:
        sys.exit(97)
else:
    sys.exit(98)
"""


def bash_executable():
    executable = shutil.which("bash")
    assert executable is not None and Path(executable).is_absolute()
    return executable


@pytest.fixture
def install_cmake(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tmp_path / "public-key").write_bytes(KEY)
    for name in ("cmake", "grep", "wget", "gpg", "sudo"):
        command = tools / name
        command.write_text("#!" + sys.executable + "\n" + COMMAND)
        command.chmod(0o755)
    driver = tmp_path / "run-cmake-installer.sh"
    driver.write_text('source "$1"\nOS=debian\nPKG_MANAGER=apt\ninstall_cmake\n')

    def run(failure=""):
        env = dict(
            os.environ,
            PATH=str(tools) + os.pathsep + os.environ["PATH"],
            CMAKE_SETUP_FIXTURE=str(tmp_path),
            CMAKE_SETUP_FAILURE=failure,
        )
        result = subprocess.run(
            [bash_executable(), str(driver), str(INSTALLER)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        calls = [
            json.loads(line)
            for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        ]
        return result, calls, tmp_path

    return run


def test_ubuntu_repository_receives_key_and_source(install_cmake):
    result, calls, root = install_cmake()
    assert result.returncode == 0, result.stderr
    assert (root / "gpg-input").read_bytes() == KEY
    assert (root / "keyring").read_bytes() == KEYRING
    assert (root / "repository").read_bytes() == REPOSITORY
    assert [call for call in calls if call[0] == "wget"] == [
        ["wget", "-O", "-", "https://apt.kitware.com/keys/kitware-archive-latest.asc"]
    ]
    assert [call for call in calls if call[:2] == ["sudo", "apt"]] == [
        ["sudo", "apt", "update"],
        ["sudo", "apt", "install", "-y", "cmake", "make"],
    ]


@pytest.mark.parametrize(
    "failure,status", [("dearmor", 11), ("keyring", 13), ("repository", 17)]
)
def test_repository_failure_stops_before_package_install(
    install_cmake, failure, status
):
    result, calls, _ = install_cmake(failure)
    assert result.returncode == status
    assert not any(call[:2] == ["sudo", "apt"] for call in calls)


def test_failed_key_download_stops_before_package_install(install_cmake):
    result, calls, _ = install_cmake("download")
    assert result.returncode != 0
    assert not any(call[:2] == ["sudo", "apt"] for call in calls)


@pytest.mark.parametrize("failure,status", [("update", 19), ("install", 23)])
def test_package_manager_failure_is_preserved(install_cmake, failure, status):
    result, _, _ = install_cmake(failure)
    assert result.returncode == status
    assert "Exit code: " + str(status) in result.stdout


def test_installed_cmake_skips_repository_and_package_changes(install_cmake):
    result, calls, _ = install_cmake("already_installed")
    assert result.returncode == 0
    assert calls == [["cmake", "--version"]]


def test_other_debian_distribution_uses_existing_package_source(install_cmake):
    result, calls, _ = install_cmake("other_debian")
    assert result.returncode == 0
    assert not any(
        call[0] in ("wget", "gpg") or call[:2] == ["sudo", "tee"] for call in calls
    )
    assert [call for call in calls if call[:2] == ["sudo", "apt"]] == [
        ["sudo", "apt", "update"],
        ["sudo", "apt", "install", "-y", "cmake", "make"],
    ]
