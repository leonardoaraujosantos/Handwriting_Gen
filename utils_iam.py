import numpy as np
import cairocffi as cairo
import torch
from matplotlib import pyplot
import matplotlib
import tqdm
import glob
import os
import lxml.etree
import h5py


def process_set(theset, iam_dir):
    blacklist = {"a08-551z-08","a08-551z-09"} # these seem broken
    txts = {}
    for i,n in enumerate(theset):
        txtfn = iam_dir+"/ascii/{}/{}/{}.txt".format(n[:3], n[:7], n)
        for j,t in enumerate(open(txtfn).read().split("CSR:")[1].strip().split("\n")):
            txts['{}-{:02d}'.format(n,j+1)] = t
    samples = {}
    for i,n in tqdm.tqdm(enumerate(theset),total=len(theset)):
        globmask = iam_dir+"/lineStrokes/{}/{}/{}-*.xml".format(n[:3], n[:7], n)
        for fn in glob.glob(globmask):
            key = (os.path.splitext(os.path.basename(fn))[0])
            if key not in blacklist:
                root = lxml.etree.parse(fn).getroot()
                strokes_list = []
                for s in root.find("StrokeSet").findall("Stroke"):
                    pts = torch.FloatTensor([(float(p.attrib['x']), float(p.attrib['y']),0.0) for p in s.findall("Point")])
                    pts[-1,-1] = 1
                    strokes_list.append(pts)
                strokes = torch.cat(strokes_list, dim=0)
                (min_x,min_y),_ = torch.min(strokes[:,:2],dim=0)
                (max_x,max_y),_ = torch.max(strokes[:,:2],dim=0)
                rel_strokes = torch.cat([torch.zeros(1,3),
                                         torch.cat([strokes[1:,:2]-strokes[:-1,:2],strokes[1:,2:]],dim=1)], dim=0)
                if rel_strokes.abs().max()<1000: # we assume that such large moves are broken
                    samples[key] = {"strokes":strokes_list, "minmax":(min_x,min_y,max_x,max_y), "rel_strokes":rel_strokes,
                                    "txt": txts[key]}
    return samples

def show_stroke(x, colors=None):   
    x= x[:(torch.arange(0,x.size(0))[x[:,2]>-0.0001].size(0))] # only used bits
    #stroke = (x[:,:2]*train_ds.coord_std.unsqueeze(0)+train_ds.coord_mean.unsqueeze(0)).cumsum(0)
    stroke = (x[:,:2]).cumsum(0)
    stroke[:,1] *= -1
    pen = x[:,2]
    xmin,ymin = stroke.min(0)[0]
    xmax,ymax = stroke.max(0)[0]
    
    actions = [matplotlib.path.Path.MOVETO]
    coords = []
    for c,p in zip(stroke, pen):
        if p >=-0.0001:
            if p==1 or len(actions)==0:
                actions.append(matplotlib.path.Path.MOVETO)
            else:
                actions.append(matplotlib.path.Path.LINETO)
            coords.append((c[0],c[1]))
    actions = actions[:-1]
    ax = pyplot.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if colors is None:
        path = matplotlib.path.Path(coords, actions)
        patch = matplotlib.patches.PathPatch(path, facecolor='none')
        ax.add_patch(patch)
    else:
        pos = coords[0]
        curpos = pos
        for pos,a,col in zip(coords, actions, colors):
            if a == matplotlib.path.Path.LINETO:
                ax.add_line(matplotlib.lines.Line2D((curpos[0],pos[0]), (curpos[1],pos[1]), axes=ax, color=col))
            curpos = pos


def stroke_to_image(x, target_size = (1280,64), randomize=False):
    #stroke = (x[:,:2]*train_ds.coord_std.unsqueeze(0)+train_ds.coord_mean.unsqueeze(0)).cumsum(0)
    stroke = (x[:,:2]).cumsum(0)
    pen = x[:,2]
    if randomize:
        shear_prob = 0.5
        shear_prec = 4.0
        if torch.rand(1)[0] > shear_prob:
            shear_sigma = 1/shear_prec**0.5
            shear_theta = 0.5*torch.randn(1)    
        else:
            shear_theta = torch.zeros(1)
        rot_prob = 0.5
        rot_prec = 4.0
        if torch.rand(1)[0] > rot_prob:
            rot_sigma = 1/rot_prec**0.5
            rot_theta = 0.5*torch.randn(1)    
        else:
            rot_theta = torch.zeros(1)

        (min_x,min_y),_ = torch.min(stroke[:,:2],dim=0)
        (max_x,max_y),_ = torch.max(stroke[:,:2],dim=0)
        stroke[:,0] -= min_x
        stroke[:,1] -= min_y  
        min_x, min_y = 0.0,0.0
        max_x, max_y = max_x-min_x, max_y-min_y

        stroke[:,0] += stroke[:,1]*torch.sin(shear_theta)

        stroke[:,0] = stroke[:,0]*torch.cos(rot_theta)+stroke[:,1]*torch.sin(rot_theta)
        stroke[:,1] = stroke[:,1]*torch.cos(rot_theta)-stroke[:,0]*torch.sin(rot_theta)

    (min_x,min_y),_ = torch.min(stroke[:,:2],dim=0)
    (max_x,max_y),_ = torch.max(stroke[:,:2],dim=0)
    stroke[:,0] -= min_x
    stroke[:,1] -= min_y  
    min_x, min_y = 0.0,0.0
    max_x, max_y = max_x-min_x, max_y-min_y

    factor = min(target_size[0]/max(max_x-min_x,0.001),target_size[1]/max(max_y-min_y,0.001),1)
    xmin,ymin = stroke.min(0)[0]
    xmax,ymax = stroke.max(0)[0]

    imwidth, imheight = int(xmax*factor)+2, int(ymax*factor)+2
    surface = cairo.ImageSurface (cairo.FORMAT_A8, imwidth, imheight)
    ctx = cairo.Context(surface)

    ctx.scale (factor, factor) # Normalizing the canvas
    ctx.rectangle (0, 0, xmax+5/factor, ymax+5/factor) # Rectangle(x0, y0, x1, y1)
    ctx.set_source_rgba (0.0, 0.0, 0.0, 0.0) # Solid color
    ctx.fill ()
    next_action = 1
    coords = []
    for c,p in zip(stroke, pen):
        if p >=-0.0001:
            if next_action:
                ctx.move_to(c[0]+1,c[1]+1)
            else:
                ctx.line_to(c[0]+1,c[1]+1)
            next_action = p>0.5
    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0) # Solid color

    
    if randomize:
        linewidth = (1+torch.rand(1)[0])/factor
    else:
        linewidth = 2/factor
    ctx.set_line_width (linewidth)
    ctx.stroke ()

    buf = surface.get_data()
    data = np.ndarray(shape=(imheight, (imwidth+3)//4*4),#(WIDTH+7)//8),
                         dtype=np.uint8,
                         buffer=buf)[:,:imwidth]
    data = 1-(data>0)
    data = torch.FloatTensor(data)
    return data


def save_to_hdf5(fn, d):
    f = h5py.File(fn, "w")
    for k,v in d.items():
        f.create_dataset(k, data=v)

def load_from_hdf5(fn):
    f = h5py.File(fn, "r")
    print (list(f.keys()))
    return {k:v[:] for k,v in f.items()}