import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from types import SimpleNamespace
from satellite_rl.tsac_search27.workbench import handler

class HTTPTests(unittest.TestCase):
    def test_page_detail_json_and_origin_guard(self):
        calls=[]
        w=SimpleNamespace(status=lambda: {'tasks':{},'gate':{'hold':False,'epoch':'e'}},
                          detail=lambda key:{'config':{'frozen':True},'owner':None,'log_tail':'small','models':'model/path'},
                          start=lambda key:calls.append(key) or {'stage':'starting'})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(w)); thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            self.assertIn('Workbench',urllib.request.urlopen(base+'/').read().decode())
            self.assertEqual(json.loads(urllib.request.urlopen(base+'/B').read())['models'],'model/path')
            for headers in [{}, {'Content-Type':'application/json','Origin':'http://other.example'}]:
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(urllib.request.Request(base+'/B/start',data=b'{}',headers=headers))
            self.assertEqual(calls,[])
            r=urllib.request.urlopen(urllib.request.Request(base+'/B/start',data=b'{}',headers={'Content-Type':'application/json','Origin':base}))
            self.assertEqual(json.loads(r.read())['stage'],'starting');self.assertEqual(calls,['B'])
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
