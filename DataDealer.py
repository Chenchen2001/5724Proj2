
def readData(file_name):
  """
    Get and process data from the server named in the url. For CMSC5724 Perceptron Project only.

    Args:
      file_name: The name of the file to read.

    Returns:
      input: The vectors read from the file.
      label: The label of each vector.
      dim: The dimension of the vectors.
      rad: The radius of these points
  """
  input = []
  label = []
  file = open(file_name, 'r')
  file_content = file.readlines()

  dataIntro = file_content[0].split(',')
  dataIntro[-1] = dataIntro[-1].replace('\n','')
  dim, rad = int(dataIntro[0]), int(dataIntro[-1])

  for row in file_content[1:]:
    list = row.split(',')
    list[-1] = list[-1].replace('\n','')
    input.append(list[0:dim])
    label.append(int(list[-1]))

  return input, label, dim, rad