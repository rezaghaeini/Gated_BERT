import math
import matplotlib
import numpy as np
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

cmapList = ['seismic', 'Purples', 'Greys', 'Blues', 'Reds', 'PuRd', 'RdPu', 'binary']

def norm2(v, ax=None):
	nv = np.linalg.norm(v, axis=ax)
	return nv

def abs_max(v, ax=None):
	v = np.abs(v)
	nv = np.max(v, axis=ax)
	return nv

def abs_sum(v, ax=None):
	v = np.abs(v)
	nv = np.sum(v, axis=ax)
	return nv

def abs_mean(v, ax=None):
	v = np.abs(v)
	nv = np.mean(v, axis=ax)
	return nv

def normalization(v, ax=None, zero_one=True, doAbs=False):
	if v is None:
		print("array is None!")
		return
	v = np.array(v, dtype='float32')
	assert len(v.shape) <= 2
	if doAbs:
		v = np.abs(v)
	max_value = np.max(v, axis=ax)
	min_value = np.min(v, axis=ax)
	if ax is None:
		v = (v - min_value) / (max_value - min_value)
	elif ax == 1 or ax == -1:
		v = (v - min_value[:,None]) / (max_value - min_value)[:,None]
	else:
		v = (v - min_value[None,:]) / (max_value - min_value)[None,:]
	if not zero_one:
		v = (2 * v) - 1
	return v

# ------------------------------------------------------------------------------------------------------------------------
def draw_heatmap(v, x=None, y=None, xlbl='', ylbl='', title=None, colorbar=False, doNormal=True,
				 normAx=None, doAbs=True, zero_one=True, cmap='Purples'):
	assert y is None or len(y) == v.shape[0]
	assert x is None or len(x) == v.shape[1]
	if doNormal:
		v = normalization(v, ax=normAx, doAbs=doAbs, zero_one=zero_one)
	im = plt.imshow(v, cmap=cmap, interpolation='nearest', aspect='auto',
					origin='lower', vmin=np.min(v), vmax=np.max(v))
	plt.yticks(np.arange(len(y)), y, rotation=45) if y is not None else plt.yticks(np.arange(v.shape[0]), ['' for _ in range(v.shape[0])])
	plt.xticks(np.arange(len(x)), x, rotation=45) if x is not None else plt.xticks(np.arange(v.shape[1]), ['' for _ in range(v.shape[1])])
	if title is not None:
		plt.title(title)
	plt.ylabel(ylbl)
	plt.xlabel(xlbl)
	if colorbar:
		plt.colorbar()

	return im

def plot(vs, xs=None, ys=None, xlbl='', ylbl='', subTitle=None, lblType='boarder', singleColorbar=True, colorbar=None,
		 doNormal=True, normAx=None, doAbs=True, zero_one=True, cmap='Purples', doVis=False, save_name=None,
		 figsize=None, nrows=None, ncols=None, figH=4.5, figW=4.5):
	vs = vs if type(vs) == list else [vs]
	if xs is None or (type(xs)==list and len(xs)>0 and type(xs[0])!=list):
		xs = [xs]
	if ys is None or (type(ys)==list and len(ys)>0 and type(ys[0])!=list):
		ys = [ys]
	xlbl = xlbl if type(xlbl) == list else [xlbl]
	ylbl = ylbl if type(ylbl) == list else [ylbl]
	doAbs = doAbs if type(doAbs) == list else [doAbs]
	colorbar = colorbar if type(colorbar) == list else [colorbar]
	doNormal = doNormal if type(doNormal) == list else [doNormal]
	normAx = normAx if type(normAx) == list else [normAx]
	cmap = cmap if type(cmap) == list else [cmap]
	zero_one = zero_one if type(zero_one) == list else [zero_one]
	subTitle = subTitle if type(subTitle) == list else [subTitle]
	vCount = len(vs)
	for i in range(len(vs)):
		if (vs[i].ndim) == 1:
			vs[i] = np.array([vs[i]])
		if vs[i].ndim > 2:
			raise RuntimeError('Unsupported vector')
	assert xs[0] is None or len(xs[0]) == vs[0].shape[1]
	assert ys[0] is None or len(ys[0]) == vs[0].shape[0]
	assert len(xs) == 1 or len(xs) == vCount
	assert len(ys) == 1 or len(ys) == vCount
	assert len(xlbl) == 1 or len(xlbl) == vCount
	assert len(ylbl) == 1 or len(ylbl) == vCount
	assert len(doAbs) == 1 or len(doAbs) == vCount
	assert len(colorbar) == 1 or len(colorbar) == vCount
	assert len(doNormal) == 1 or len(doNormal) == vCount
	assert len(normAx) == 1 or len(normAx) == vCount
	assert len(cmap) == 1 or len(cmap) == vCount
	assert len(zero_one) == 1 or len(zero_one) == vCount
	assert len(subTitle) == 1 or len(subTitle) == vCount

	if vCount > 1:
		if (nrows is not None and nrows > 0) and (ncols is not None) and ncols > 0:
			pass
		elif (nrows is None or nrows <= 0) and (ncols is not None) and ncols > 0:
			assert vCount % ncols == 0
			nrows = vCount/ncols
		elif (ncols is None or ncols <= 0) and (nrows is not None) and nrows > 0:
			assert vCount % nrows == 0
			ncols = vCount/nrows
		else:
			nrows = 1
			ncols = vCount

		if lblType == 'boarder':
			xs = [xs[0] if i>=ncols*(nrows-1) else None for i in range(vCount)]
			ys = [ys[0] if i%ncols==0 else None for i in range(vCount)]
			xlbl = [xlbl[0] if i>=ncols*(nrows-1) else '' for i in range(vCount)]
			ylbl = [ylbl[0] if i%ncols==0 else '' for i in range(vCount)]
	else:
		nrows = 1
		ncols = 1

	top_title = 0 if subTitle[0] is None else 1
	middle_title = 0 if (subTitle[0] is None or subTitle[ncols] is None) else 1
	bottom_x_max = 0 if xs[0] is None else max([len(x) for x in xs[0]])
	left_y_max = 0 if ys[0] is None else max([len(y) for y in ys[0]])
	if lblType == 'boarder' or xs[0] is None or vCount==1:
		middle_x_max = 0
	else:
		middle_x_max = bottom_x_max if len(xs)==1 else max([len(x) for x in xs[1]])
	if lblType == 'boarder' or ys[0] is None or vCount==1:
		middle_y_max = 0
	else:
		middle_y_max = left_y_max if len(ys)==1 else max([len(y) for y in ys[1]])
	xlblMarg = 0 if xlbl[-1] is None else 1
	ylblMarg = 0 if ylbl[0] is None else 1
	xlblMidMarg = 0 if lblType == 'boarder' or xlbl[-1] is None or vCount==0 or xlbl[0] is None else 1
	ylblMidMarg = 0 if lblType == 'boarder' or ylbl[-1] is None or vCount==0 or ylbl[0] is None else 1

	plt.cla()
	plt.clf()
	plt.close('all')
	if figsize is None:
		figsize = (figW*ncols+0.8,figH*nrows)
	fig = plt.figure(figsize=figsize)

	leftMarg = ((.2*ylblMarg+.2+0.1*left_y_max/math.sqrt(2)) / figsize[0])
	bottomMarg = ((.2*xlblMarg+.3+0.1*bottom_x_max/math.sqrt(2)) / figsize[1])
	rightMarg = 1-(.4 / figsize[0])
	topMarg = 1-((.4+top_title*0.1) / figsize[1])
	wMarg = ((.7*ylblMidMarg+.4+0.3*middle_x_max/math.sqrt(2)) / figsize[0])
	hMarg = ((.6*xlblMidMarg+.4+0.3*middle_y_max/math.sqrt(2)+middle_title*0.7) / figsize[1])

	if vCount > 1:
		plt.subplots_adjust(left=leftMarg, bottom=bottomMarg, right=rightMarg, top=topMarg, wspace=wMarg, hspace=hMarg)

	for i in range(vCount):
		if vCount > 1:
			plt.subplot(100*nrows+10*ncols+i+1)
		im = draw_heatmap(vs[i],
					 x=xs[0] if len(xs)==1 else xs[i],
					 y=ys[0] if len(ys)==1 else ys[i],
					 title=subTitle[0] if len(subTitle)==1 else subTitle[i],
					 xlbl=xlbl[0] if len(xlbl)==1 else xlbl[i],
					 ylbl=ylbl[0] if len(ylbl)==1 else ylbl[i],
					 colorbar=(not singleColorbar) and (colorbar[0] if len(colorbar)==1 else colorbar[i]),
					 doNormal=doNormal[0] if len(doNormal)==1 else doNormal[i],
					 normAx=normAx[0] if len(normAx)==1 else normAx[i],
					 doAbs=doAbs[0] if len(doAbs)==1 else doAbs[i],
					 zero_one=zero_one[0] if len(zero_one)==1 else zero_one[i],
					 cmap=cmap[0] if len(cmap)==1 else cmap[i])

	if singleColorbar:
		rightSubMarg = .8 / figsize[0]
		fig.subplots_adjust(right=1-rightSubMarg)
		bar_width = .2 / figsize[0]
		cbar_ax = fig.add_axes([1-3*bar_width, 0.11, bar_width, 0.78])
		fig.colorbar(im, cax=cbar_ax)

	if doVis:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, bbox_inches='tight')


def range_plot(vs, save_name, x=None, y=None, xlbl='', ylbl='',
				colorbar=None, normAx=None, doAbs=True,
				cmap='Purples', figsize=None, figH=4.5, figW=4.5):
	vs = vs if type(vs) == list else [vs]
	save_name = save_name if type(save_name) == list else [save_name]
	doAbs = doAbs if type(doAbs) == list else [doAbs]
	normAx = normAx if type(normAx) == list else [normAx]

	vCount = len(vs)
	for i in range(len(vs)):
		if (vs[i].ndim) == 1:
			vs[i] = np.array([vs[i]])
		if vs[i].ndim > 2:
			raise RuntimeError('Unsupported vector')
	assert len(vs) == len(save_name)
	assert x is None or len(x) == vs[0].shape[1]
	assert y is None or len(y) == vs[0].shape[0]
	assert len(doAbs) == 1 or len(doAbs) == vCount
	assert len(normAx) == 1 or len(normAx) == vCount

	bottom_x_max = 0 if x is None else len(x)
	left_y_max = 0 if y is None else len(y)
	xlblMarg = 0 if xlbl is None else 1
	ylblMarg = 0 if ylbl is None else 1

	plt.cla()
	plt.clf()
	plt.close('all')
	if figsize is None:
		figsize = (figW+0.8,figH)
	fig = plt.figure(figsize=figsize)

	leftMarg = ((.2*ylblMarg+.2+0.1*left_y_max/math.sqrt(2)) / figsize[0])
	bottomMarg = ((.2*xlblMarg+.3+0.1*bottom_x_max/math.sqrt(2)) / figsize[1])
	rightMarg = 1-(.4 / figsize[0])
	topMarg = 0.98

	v = normalization(vs[0], ax=normAx[0], doAbs=doAbs[0], zero_one=True)
	im = plt.imshow(v, cmap=cmap, interpolation='nearest', aspect='auto',
					origin='lower', vmin=0, vmax=1)
	plt.yticks(np.arange(len(y)), y, rotation=45) if y is not None else plt.yticks(np.arange(v.shape[0]), ['' for _ in range(v.shape[0])])
	plt.xticks(np.arange(len(x)), x, rotation=45) if x is not None else plt.xticks(np.arange(v.shape[1]), ['' for _ in range(v.shape[1])])
	plt.ylabel(ylbl)
	plt.xlabel(xlbl)
	if colorbar:
		plt.colorbar()

	plt.savefig(save_name[0], bbox_inches='tight')

	for i in range(1, vCount):
		v = normalization(vs[i],
					ax=(normAx[i] if len(normAx)>1 else normAx[0]),
					doAbs=(doAbs[i] if len(doAbs)>1 else doAbs[0]), 
					zero_one=True)
		im.set_data(v)

		plt.savefig(save_name[i], bbox_inches='tight')

 # ------------------------------------------------------------------------------------------------------------------------
