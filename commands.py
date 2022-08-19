import docker
client = docker.from_env()
client.containers.run("lukasblecher/pix2tex:api", detach=True, ports={'8502/tcp': 8502})
client = docker.from_env()
for container in client.containers.list():
    container.stop()
