import os
import numpy as np
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('Deploy/templates/'))

def generate_(template_name, feature_obj):
    template = env.get_template(template_name+'.jinja')
    header_text= template.render(feature_obj=feature_obj)

    outfile_path=os.path.join("Deploy/headers/"+template_name)
    header_file = open(outfile_path, "w")
    header_file.write(header_text)
    header_file.close()