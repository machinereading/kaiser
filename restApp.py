
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append('../')

from flask import Flask
from flask_restful import Api, Resource, reqparse

from KAIST_frame_parser import srl_based_parser
from FRDF import frdf
from FRDF import kotimex
from pprint import pprint


# In[2]:


import jpype
jpype.attachThreadToJVM()


# In[3]:


app = Flask(__name__)
api = Api(app)


# In[1]:


class FrameNetRE(Resource):
    def __init__(self):
#         model_dir = '/disk_4/resource/models'
        model_dir = '/home/hahmyg/FrameNet-RE/models'
        self.FRDF = frdf.frame2RDF()
        self.fnparser = srl_based_parser.SRLbasedParser(model_dir=model_dir)
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type=str)
            parser.add_argument('sentence_id', type=str)
            args = parser.parse_args()
            print(args)
            if not args['sentence_id']:
                sentence_id = 'input_sent'
            else:
                sentence_id = args['sentence_id']
            parsed = self.fnparser.parser(args['text'], sentence_id=sentence_id)
            triples = self.FRDF.frame2dbo(parsed['conll'], sentence_id=sentence_id)
            
            result = {}
            result['textae'] = parsed['textae']
            result['frdf'] = triples
            result['conll'] = parsed['conll']
            return result, 200
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {'error':str(e)}
        


# In[10]:


api.add_resource(FrameNetRE, '/FRDF')
# api.add_resource(FrameNetRE, '/FRDF/')
app.run(debug=True, host='0.0.0.0', port=1106)

