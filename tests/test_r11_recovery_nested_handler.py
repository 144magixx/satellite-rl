"""Real nested Handler.dispatch via fake HTTPServer. No sockets, thread or child."""
import contextlib,importlib.util,io,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r10 import contracts as c
PR=dict(path='plan',sha256='p');RR=dict(path='review',sha256='r')
class Thread:
 def __init__(self,target,**kw):self.target=target
 def start(self):pass
class NestedHandlerChecks(unittest.TestCase):
 def full(self,actions=('POST',),wrongref=False,nonmain=False,deadline=300,stale=False,pause=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=dict(run_id='r11_same',request_id='same_request',archive_root=str(root),design_ref={},workbench_port=9050);calls=[];statuses=[];responses=[];tid_calls=[0];validates=[0]
   def fakepopen(cmd,**kw):calls.append((cmd,kw));return types.SimpleNamespace(pid=888)
   def tid():
    tid_calls[0]+=1;return os.getpid()+1 if nonmain and tid_calls[0]>1 else os.getpid()
   def validate(*args):
    validates[0]+=1
    if stale and validates[0]>1:raise ValueError('stale recovery epoch')
    return p
   class HTTP:
    def __init__(self,address,handler):self.handler=handler
    def serve_forever(self):
     for action in actions:
      obj=self.handler.__new__(self.handler);obj.command='GET' if action=='GET' else 'POST';obj.path='/api/ai-search-r11/tasks/r11_same'+('/stop' if action=='STOP' else '/start' if action=='POST' else '');body=dict(plan_ref={'wrong':1} if wrongref else PR,root_review_ref=RR,request_id='same_request');raw=json.dumps(body).encode();obj.headers={'Content-Length':str(len(raw))};obj.rfile=io.BytesIO(raw);obj.wfile=io.BytesIO();obj.send_response=lambda code:statuses.append(code);obj.send_header=lambda *a:None;obj.end_headers=lambda:None;obj.dispatch();responses.append(json.loads(obj.wfile.getvalue()))
   name='scripts.stage1_ai_search_r11_prospective.server_full_handler_under_test';path=Path(__file__).parents[1]/'scripts/stage1_ai_search_r11_prospective/server.py'
   with patch('subprocess.Popen',fakepopen):
    spec=importlib.util.spec_from_file_location(name,path);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
   with patch.object(s,'HTTPServer',HTTP),patch.object(s,'validate',side_effect=validate),patch.object(s,'check_pause',side_effect=RuntimeError('pause') if pause else None),patch.object(s,'fresh_route'),patch.object(s.threading,'get_native_id',side_effect=tid),patch.object(s.threading,'Thread',Thread),patch.object(s,'boottime',return_value=100),patch.object(s.c,'identity',side_effect=lambda pid:dict(pid=pid,birth_ticks=1,boot_id='b',uid=500902)),patch.dict(os.environ,{'R11_LANE_DEADLINE_BOOTTIME':str(deadline)}),patch.object(sys,'argv',['server','--plan','plan','--plan-sha256','p','--review','review','--review-sha256','r','--port','9050']):
    try:s.main()
    except Exception as e:responses.append(dict(main_error=str(e)))
   files={x.name:json.loads(x.read_bytes()) for x in root.iterdir() if x.name.endswith('.json') or x.name=='stop.request'};return calls,statuses,responses,files
 def test_real_nested_main_handler_one_spawn_and_persistent_reaper(self):
  calls,statuses,responses,files=self.full();self.assertEqual(statuses,[200]);self.assertEqual(len(calls),1);self.assertFalse(responses[0]['POST_replayed']);self.assertIn('actual_worker_launch.json',files);self.assertTrue(callable(calls[0][1]['preexec_fn']))
 def test_nonmain_handler_never_intent_or_spawn(self):
  calls,statuses,_,files=self.full(nonmain=True);self.assertEqual(calls,[]);self.assertEqual(statuses,[409]);self.assertNotIn('submission_intent.json',files)
 def test_duplicate_post_never_second_spawn(self):
  calls,statuses,_,_=self.full(actions=('POST','POST'));self.assertEqual(len(calls),1);self.assertEqual(statuses,[200,409])
 def test_wrongref_no_spawn(self):
  calls,statuses,_,_=self.full(wrongref=True);self.assertEqual(calls,[]);self.assertEqual(statuses,[409])
 def test_stale_post_no_spawn(self):
  calls,statuses,_,_=self.full(stale=True);self.assertEqual(calls,[]);self.assertEqual(statuses,[409])
 def test_expired_deadline_no_spawn(self):
  calls,statuses,_,_=self.full(deadline=100);self.assertEqual(calls,[]);self.assertEqual(statuses,[409])
 def test_stop_then_get_no_spawn(self):
  calls,statuses,_,files=self.full(actions=('STOP','GET'));self.assertEqual(calls,[]);self.assertEqual(statuses,[200,200]);self.assertIn('stop.request',files)
 def test_pause_no_spawn(self):
  calls,_,_,_=self.full(pause=True);self.assertEqual(calls,[])
if __name__=='__main__':unittest.main()
