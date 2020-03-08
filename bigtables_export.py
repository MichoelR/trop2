import os, sys, collections
from tf.fabric import Fabric

# locations = '~/github/etcbc'
locations = '/home/oem/text-fabric-data/etcbc'
coreModule = 'bhsa'
sources = [coreModule, 'phono']
# version = '2017'
version = 'c'
tempDir = os.path.expanduser(f'{locations}/{coreModule}/_temp/{version}/r')
tableFile = f'{tempDir}/{coreModule}{version}.txt'

modules = [f'{s}/tf/{version}' for s in sources]
TF = Fabric(locations=locations, modules=modules)

api = TF.load('')
api = TF.load(('suffix_person',
 'tab',
 'trailer',
 'trailer_utf8',
 'txt',
 'typ',
 'uvf',
 'vbe',
 'vbs',
 'verse',
 'voc_lex',
 'voc_lex_utf8',
 'vs',
 'vt',
 'distributional_parent',
 'functional_parent',
 'mother',
 'oslots'))
allFeatures = TF.explore(silent=False, show=True)
loadableFeatures = allFeatures['nodes'] + allFeatures['edges']
del(api)
api = TF.load(loadableFeatures)
api.makeAvailableIn(globals())

print('done')
