import requests
def download_file_url(file_url,filename):

    # download file
    r = requests.get(file_url, stream = True) 

    with open(filename,"wb") as f: 
        for chunk in r.iter_content(chunk_size=1024): 

             # writing one chunk at a time to pdf file 
             if chunk: 
                 f.write(chunk)
    return()

parent_dir = './Data/'
file_names = ['gambier.ubc', 'beadpack.ubc', 'sandpack.ubc', 'castlegate.ubc']
file_dirs = [parent_dir+file for file in file_names]
file_links = ['https://www.digitalrocksportal.org/projects/16/images/65565/download/',
              'https://www.digitalrocksportal.org/projects/16/images/65563/download/',
              'https://www.digitalrocksportal.org/projects/16/images/65566/download/',
              'https://www.digitalrocksportal.org/projects/16/images/65564/download/']

for i in range(0,len(file_dirs)):
    download_file_url(file_links[i],file_dirs[i])

