import re
import json

import os
import re

def readvtk(vtkfile):
    with open(vtkfile, "r") as f:
        content = "".join([line for line in f])
        lines = [each.strip() for each in content.split("SCALARS") if each.strip()]
        dimensions = [int(each) for each in re.findall(r"DIMENSIONS (\d+) (\d+) (\d*)", lines[0])[0]]
        parts = [line.split("LOOKUP_TABLE default") for line in lines[1:]]
        # print(parts)
        labels = [part[0].strip().split()[0] for part in parts]
        datas = [[float(each.strip()) for each in part[1].strip().split() if each.strip()] for part in parts]
        # print(labels)
        # print(datas)
        return dict(zip(labels, datas)), dimensions

def writevtk(vtkfile, data):
    template = """# vtk DataFile Version 3.0 
PhaseField 
ASCII 
DATASET STRUCTURED_POINTS 
DIMENSIONS {Nx} {Ny} {Nz}
ASPECT_RATIO 1.0 1.0 1.0 
ORIGIN 0.0 0.0 0.0 

POINT_DATA {size}
SCALARS Conc double
LOOKUP_TABLE default
{data}
"""
    with open(vtkfile, "w") as f:
        f.write(template.format(Nx=data["Nx"], Ny=data["Ny"], Nz=data["Nz"], size=data["size"], data=data["data"]))

if __name__ == "__main__":
    for var, block in readvtk(r"C:\Users\zhong\Desktop\statistics_vtks\b-0.2875\conc_1800000s.vtk"):
        print(var)
        print(block)
