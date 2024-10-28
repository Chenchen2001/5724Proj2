from urllib import request

def readData(url: str) -> tuple:
  """
    Get and process data from the server named in the url. For CMSC5724 Perceptron Project only.

    Args:
      url: the url of remote resource

    Returns:
      A tuple of dataIntro and dataContent,.
  """
  data = request.urlopen(url)
  data = data.read().decode('utf-8')
  data = [[float(item) for item in line.split(",")] for line in data.split("\n")[:-1]]
  dataIntro = data[0]
  dataContent = data[1:]
  return (dataIntro, dataContent)