"""Independent bounded whole-byte verification and same-owner transport receipt."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for body in iter(lambda:stream.read(8*1024**2),b''):h.update(body)
    return h.hexdigest()

def main():
    ticket_path=Path(sys.argv[1]);expected=sys.argv[2]
    if sha(ticket_path)!=expected:raise ValueError('exact raw transport ticket required')
    ticket=json.loads(ticket_path.read_text());root=Path(ticket['canonical_directory']).resolve()
    if str(root)!='/home/u2024110224/second_stage_satellite_rl/s2_w26/shared_data/official':raise ValueError('canonical containment')
    deadline=time.monotonic()+14400
    while not (root/'transport_aggregate.json').exists():
        if (root/'transport_failure.json').exists():raise RuntimeError('raw transport failure retained; no receipt pass')
        if time.monotonic()>deadline:raise TimeoutError('bounded raw verification wait elapsed')
        time.sleep(5)
    aggregate=json.loads((root/'transport_aggregate.json').read_text())
    if aggregate['status']!='transport_complete':raise ValueError('both raw originals must be completed')
    entries=[];owner_proofs={};transport=[]
    for row in ticket['assets']:
        day=row['date'];path=root/(day+'.tar');owner=root/(day+'.owner.json')
        identity=dict(request_id=ticket['request_id'],source_admission_SHA=ticket['original_download_admission_SHA'],
             source_url=row['source_url'],release_asset_id=row['asset_id'],release_asset_bytes=row['expected_bytes'])
        if json.loads(owner.read_text())!=identity:raise ValueError('same original task owner differs')
        if day=='2024-02-23' and owner.read_bytes()!=json.dumps(identity,sort_keys=True,allow_nan=False).encode():
            raise ValueError('original Feb23 frozen owner bytes changed')
        owner_proofs[day]=dict(path=str(owner),sha256=sha(owner),same_original_identity=True,
                              original_frozen_owner_bytes_preserved=day=='2024-02-23',
                              newly_created_for_same_task=day=='2024-02-24')
        status=json.loads((root/(day+'.transport_status.json')).read_text())
        ledger=json.loads((root/(day+'.transport_ledger.json')).read_text())
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['expected_bytes']:raise ValueError('canonical original size/type differs')
        actual=sha(path)
        if actual!=status['whole_SHA'] or actual!=ledger['whole_SHA']:raise ValueError('independent whole original SHA differs')
        entries.append(dict(path=str(path),bytes=path.stat().st_size,sha256=actual,source_url=row['source_url'],release_asset_id=row['asset_id'],date=day))
        transport.append(dict(date=day,transport_url=row['transport_url'],official_redirect_chain=row['redirect_chain'],
            source_API_metadata=row['metadata'],redirect_observed_UTC=row['observed_UTC'],TLS_verified=row['TLS_verified'],
            chunk_count=len(ledger['chunks']),failed_reads=ledger['failures'],completed_UTC=ledger['completed_UTC']))
    value=dict(schema='W26OfficialTransportReceiptV1',root_approved=True,passed=True,assets=entries,
        request_id=ticket['request_id'],original_run_id=ticket['original_run_id'],
        original_download_admission_SHA=ticket['original_download_admission_SHA'],ticket_SHA=expected,
        owner_proofs=owner_proofs,transport=transport,no_trace_read=True,algorithm_calls=0,
        observed_UTC=datetime.now(timezone.utc).isoformat(),independent_full_byte_readback=True,
        original_failed_cost=dict(run_id=ticket['original_run_id'],worker_pid=3238,birth_ticks=853311151,
            started_UTC='2026-10-14T00:52:16.778857+00:00',failed_UTC='2026-10-14T00:52:47.294411+00:00',
            seconds=30.515554,error="URLError(TimeoutError('timed out'))",bytes=0,original_job_preserved=True),
        requires_original_workbench_sameID_ingest=True,root_source_publication=False)
    output=root/'official_transport_receipt_v1.json'
    with output.open('x') as stream:json.dump(value,stream,sort_keys=True)
    print(json.dumps(dict(receipt_path=str(output),sha256=sha(output),assets=entries)),flush=True)

if __name__=='__main__':main()
