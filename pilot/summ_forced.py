import pandas as pd, json, sys
rows=[]
for f,tag in [('results_forced.json','base'),('results_forced_lora.json','lora150'),('results_forced_lora250.json','lora250')]:
    try: d=json.load(open(f))
    except FileNotFoundError: continue
    for r in d:
        for w,p in r['p'].items(): r[f'p{w}']=p
        r['model']=tag; rows.append(r)
d=pd.DataFrame(rows); d['cond']=d.model+'/'+d.cond
d['ok']=d.apply(lambda r: r.top==r.correct if r.correct else None,axis=1)
v=['need_end','need_mid','ctrl_full','ctrl_end2','ctrl_end4']
print('Mean P(1x3) [among 1x1..1x4]'); print(d.pivot_table(index='cond',columns='variant',values='p3')[v].round(3))
print('\nTop choice correct % (need: 1x3; ctrl_end2: 1x2; ctrl_end4: 1x4)'); print((d.pivot_table(index='cond',columns='variant',values='ok',aggfunc='mean')*100).reindex(columns=[x for x in v if x!='ctrl_full']).round(0))
print('\nMost common top choice'); print(d.pivot_table(index='cond',columns='variant',values='top',aggfunc=lambda s:s.mode()[0])[v])
