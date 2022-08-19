#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import docker

container = None


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Rekognition.settings')
    try:
        container.stop()
    except BaseException:
        pass
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    try:

        client = docker.from_env()
        container = client.containers.run("lukasblecher/pix2tex:api", detach=True,
                                          ports={'8502/tcp': 8502})

    except BaseException:
        pass

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
