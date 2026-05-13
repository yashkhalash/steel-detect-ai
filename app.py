import os, json, uuid, random, datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = 'tata-steel-ai-2026'
ALLOWED = {'png','jpg','jpeg','bmp','tiff','webp'}

DEFECTS = {
    'crazing':         {'color':'#FF6B6B','severity':'Medium','scrap_rate':0.15,'rework_hrs':2.0,
                        'desc':'Network of fine surface cracks from thermal stress',
                        'root_causes':['Rapid cooling rate in annealing furnace','Excessive thermal gradient in hot rolling','Inadequate soaking time at normalising temperature'],
                        'propagation':{'Rolling':'Cracks widen under compressive load — surface delamination risk','Annealing':'May anneal out if shallow; deep cracks worsen with repeated cycles','Coating':'Coating adhesion failure over crack network — corrosion entry point'}},
    'inclusion':       {'color':'#FFD93D','severity':'High','scrap_rate':0.55,'rework_hrs':0.0,
                        'desc':'Non-metallic foreign material embedded in steel matrix',
                        'root_causes':['Ladle slag carryover during tapping','Inadequate argon stirring in ladle metallurgy','Refractory erosion in tundish — alumina inclusions'],
                        'propagation':{'Rolling':'Inclusion elongates into streak defect in final strip','Annealing':'No improvement; inclusion remains as stress concentrator','Coating':'Pinholes in coating directly above inclusion site'}},
    'patches':         {'color':'#6BCB77','severity':'Low','scrap_rate':0.05,'rework_hrs':1.5,
                        'desc':'Irregular discoloured surface patches from scale adhesion',
                        'root_causes':['Inconsistent descaling water pressure','Scale build-up on furnace rolls','Non-uniform heating in reheat furnace'],
                        'propagation':{'Rolling':'Patches may be rolled smooth; risk is low','Annealing':'Oxidation can worsen patch boundary contrast','Coating':'Coating thickness variation over patch area'}},
    'pitted_surface':  {'color':'#4D96FF','severity':'High','scrap_rate':0.45,'rework_hrs':0.0,
                        'desc':'Hemispherical cavities caused by entrapped gas or dissolved inclusions',
                        'root_causes':['Hydrogen content too high in liquid steel (>2 ppm)','Gas entrapment during continuous casting','Excessive moisture in mould flux'],
                        'propagation':{'Rolling':'Pits collapse into internal voids — lamination risk','Annealing':'Hydrogen embrittlement if H not fully diffused out','Coating':'Pits create coating skip — corrosion nucleation sites'}},
    'rolled_in_scale': {'color':'#FF922B','severity':'Medium','scrap_rate':0.20,'rework_hrs':3.0,
                        'desc':'Oxide scale pressed into surface during hot rolling',
                        'root_causes':['Descaler nozzle blockage or low water pressure','Excessive furnace scale from high O2 atmosphere','Insufficient inter-pass descaling in roughing mill'],
                        'propagation':{'Rolling':'Scale pressed deeper — becomes embedded metallic oxide','Annealing':'No improvement; scale fused to matrix','Coating':'Scale fragments detach — flaking in service'}},
    'scratches':       {'color':'#CC5DE8','severity':'Low','scrap_rate':0.03,'rework_hrs':1.0,
                        'desc':'Linear mechanical scratches from handling or roll contact',
                        'root_causes':['Guide groove wear in rolling mill','Damaged coiler mandrel surface','Abrasive contamination on conveyor rolls'],
                        'propagation':{'Rolling':'Scratch depth increases; work hardening at edges','Annealing':'Shallow scratches may remain as stress raisers','Coating':'Coating bridges over scratch; under-film corrosion risk'}}
}

GRADE_COST = {
    'HR (Hot Rolled)':    {'base_price':52000,'scrap_penalty':0.60,'rework_cost_hr':1200,'co2_kg_per_tonne':1850},
    'CR (Cold Rolled)':   {'base_price':68000,'scrap_penalty':0.65,'rework_cost_hr':1800,'co2_kg_per_tonne':2100},
    'Galvanised (GI/GL)': {'base_price':85000,'scrap_penalty':0.70,'rework_cost_hr':2400,'co2_kg_per_tonne':2400},
    'API Grade':          {'base_price':95000,'scrap_penalty':0.75,'rework_cost_hr':2800,'co2_kg_per_tonne':1950},
}

inspection_log = []

def allowed(f): return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED

def simulate_detection(path):
    img = Image.open(path).convert('RGB')
    w,h = img.size
    n = random.randint(0,4)
    chosen = random.sample(list(DEFECTS.keys()), min(n, len(DEFECTS)))
    dets = []
    for cls in chosen:
        x1=random.randint(0,int(w*0.55)); y1=random.randint(0,int(h*0.55))
        bw=random.randint(int(w*0.1),int(w*0.33)); bh=random.randint(int(h*0.1),int(h*0.33))
        dets.append({'class':cls,'confidence':round(random.uniform(0.72,0.98),2),
                     'bbox':[x1,y1,min(x1+bw,w-1),min(y1+bh,h-1)],
                     'area_pct':round(random.uniform(1.5,18.0),1)})
    return dets

def draw_detections(src, dets, dst):
    img = Image.open(src).convert('RGB')
    draw = ImageDraw.Draw(img,'RGBA')
    for d in dets:
        h=DEFECTS[d['class']]['color']; r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16)
        x1,y1,x2,y2=d['bbox']
        draw.rectangle([x1,y1,x2,y2],outline=(r,g,b,255),width=3)
        draw.rectangle([x1,y1,x2,y2],fill=(r,g,b,35))
        lbl=f"{d['class'].replace('_',' ').title()} {d['confidence']*100:.0f}%"
        draw.rectangle([x1,max(0,y1-20),x1+len(lbl)*7+6,y1],fill=(r,g,b,220))
        draw.text((x1+3,max(0,y1-18)),lbl,fill=(255,255,255))
    img.save(dst,quality=95)

def generate_gradcam(src, dst):
    img=Image.open(src).convert('RGB').resize((448,448))
    H,W=448,448
    x=np.linspace(0,1,W); y=np.linspace(0,1,H); xx,yy=np.meshgrid(x,y)
    heat=np.zeros((H,W),dtype=np.float32)
    for _ in range(3):
        cx,cy=random.uniform(0.2,0.8),random.uniform(0.2,0.8); s=random.uniform(0.1,0.25)
        heat+=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*s**2))
    heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8)
    cm=np.zeros((H,W,3),dtype=np.uint8)
    cm[...,0]=(heat*255).astype(np.uint8); cm[...,1]=((1-heat)*80).astype(np.uint8); cm[...,2]=50
    blended=Image.blend(img,Image.fromarray(cm,'RGB'),0.5); blended.save(dst,quality=95)

def calc_cost(dets, grade, coil_weight_t=20.0):
    if grade not in GRADE_COST: grade=list(GRADE_COST.keys())[0]
    gc=GRADE_COST[grade]; total_scrap=0; total_rework=0; total_co2=0; breakdown=[]
    for d in dets:
        df=DEFECTS[d['class']]; area=d.get('area_pct',5.0)/100.0
        sev=df['severity']
        scrap_t=coil_weight_t*area*df['scrap_rate']*(1.5 if sev=='High' else 1.0)
        scrap_cost=scrap_t*gc['base_price']*gc['scrap_penalty']
        rework_cost=df['rework_hrs']*gc['rework_cost_hr']
        co2=scrap_t*gc['co2_kg_per_tonne']
        total_scrap+=scrap_cost; total_rework+=rework_cost; total_co2+=co2
        breakdown.append({'defect':d['class'].replace('_',' ').title(),'scrap_cost':round(scrap_cost),
                          'rework_cost':round(rework_cost),'co2_kg':round(co2),'severity':sev})
    return {'total':round(total_scrap+total_rework),'scrap':round(total_scrap),'rework':round(total_rework),
            'co2_kg':round(total_co2),'breakdown':breakdown,'grade':grade}

def propagation_risk(dets):
    risks=[]
    for d in dets:
        df=DEFECTS[d['class']]
        stage_risks=[{'stage':stage,'risk':('High' if df['severity']=='High' else 'Medium'),'detail':desc}
                     for stage,desc in df['propagation'].items()]
        risks.append({'defect':d['class'].replace('_',' ').title(),'stages':stage_risks,'confidence':d['confidence']})
    return risks

def root_cause_hypotheses(dets):
    hyps=[]
    for d in dets:
        df=DEFECTS[d['class']]
        causes=[{'rank':i+1,'cause':c,'probability':max(90-i*12,30)} for i,c in enumerate(df['root_causes'][:3])]
        hyps.append({'defect':d['class'].replace('_',' ').title(),'causes':causes,'confidence':d['confidence']})
    return hyps

def generate_shift_narrative(log_slice):
    total=len(log_slice); defective=sum(1 for r in log_slice if r['defect_count']>0)
    pass_rate=round(100*(total-defective)/max(total,1),1)
    cost=sum(r.get('cost_inr',0) for r in log_slice)
    dc={}
    for r in log_slice:
        for d in r.get('defects',[]): dc[d]=dc.get(d,0)+1
    top_defect=max(dc,key=dc.get).replace('_',' ').title() if dc else 'None'
    verdict='EXCELLENT' if pass_rate>95 else ('GOOD' if pass_rate>85 else ('CONCERNING' if pass_rate>70 else 'CRITICAL'))
    narrative=(f"Shift Quality Summary — {datetime.datetime.now().strftime('%d %b %Y %H:%M')}\n\n"
               f"Production Status: {verdict}\n\n"
               f"This shift processed {total} coil inspection(s) with a quality pass rate of {pass_rate}%. "
               f"{'All coils met quality specifications — excellent process control.' if defective==0 else f'{defective} coil(s) showed surface defects requiring attention.'} "
               f"The most frequent defect type was {top_defect}. "
               f"Estimated cost-of-quality impact this shift: INR {cost:,.0f}. "
               f"{'No immediate corrective action required.' if pass_rate>90 else 'Recommend reviewing upstream process parameters — particularly descaling pressure and furnace atmosphere — before next shift.'}")
    return narrative

@app.route('/')
def index():
    stats={'total':len(inspection_log),
           'defective':sum(1 for r in inspection_log if r['defect_count']>0),
           'pass_rate':round(100*sum(1 for r in inspection_log if r['defect_count']==0)/max(len(inspection_log),1),1),
           'total_cost':sum(r.get('cost_inr',0) for r in inspection_log)}
    return render_template('index.html',stats=stats)

@app.route('/inspect',methods=['POST'])
def inspect():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    f=request.files['file']
    if not f or not allowed(f.filename): return jsonify({'error':'Invalid file type'}),400
    grade=request.form.get('grade','HR (Hot Rolled)')
    coil_id=request.form.get('coil_id',f'COIL-{random.randint(10000,99999)}')
    uid=str(uuid.uuid4())[:8]
    fname=secure_filename(f'{uid}_{f.filename}')
    upath=os.path.join(app.config['UPLOAD_FOLDER'],fname)
    f.save(upath)
    dets=simulate_detection(upath)
    ann=f'annot_{fname}'; gcam=f'gcam_{fname}'
    draw_detections(upath,dets,os.path.join(app.config['UPLOAD_FOLDER'],ann))
    generate_gradcam(upath,os.path.join(app.config['UPLOAD_FOLDER'],gcam))
    sevs=[DEFECTS[d['class']]['severity'] for d in dets]
    max_sev='High' if 'High' in sevs else ('Medium' if 'Medium' in sevs else ('Low' if sevs else 'None'))
    verdict='PASS' if not dets else ('FAIL' if max_sev=='High' else 'REVIEW')
    avg_conf=round(sum(d['confidence'] for d in dets)/max(len(dets),1),2)
    cost=calc_cost(dets,grade)
    prop=propagation_risk(dets)
    hyps=root_cause_hypotheses(dets)
    result={'id':uid,'coil_id':coil_id,'filename':f.filename,'original':fname,'annotated':ann,'gradcam':gcam,
            'detections':dets,'defect_count':len(dets),'verdict':verdict,'max_severity':max_sev,
            'avg_confidence':avg_conf,'cost':cost,'propagation':prop,'hypotheses':hyps,'grade':grade,
            'timestamp':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'defect_details':[{'class':d['class'],'display':d['class'].replace('_',' ').title(),
                               'confidence':d['confidence'],'severity':DEFECTS[d['class']]['severity'],
                               'color':DEFECTS[d['class']]['color'],'desc':DEFECTS[d['class']]['desc'],
                               'area_pct':d.get('area_pct',5.0),'bbox':d['bbox']} for d in dets]}
    inspection_log.append({'id':uid,'coil_id':coil_id,'timestamp':result['timestamp'],'filename':f.filename,
                           'defect_count':len(dets),'verdict':verdict,'avg_confidence':avg_conf,
                           'max_severity':max_sev,'grade':grade,'cost_inr':cost['total'],
                           'defects':[d['class'] for d in dets]})
    return jsonify(result)

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html',log=inspection_log[-100:])

@app.route('/mobile')
def mobile(): return render_template('mobile.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    verdicts={'PASS':0,'FAIL':0,'REVIEW':0}; defect_freq={}; total_cost=0
    for r in inspection_log:
        verdicts[r['verdict']]=verdicts.get(r['verdict'],0)+1
        total_cost+=r.get('cost_inr',0)
        for d in r.get('defects',[]): defect_freq[d]=defect_freq.get(d,0)+1
    return jsonify({'total':len(inspection_log),'verdicts':verdicts,'defect_freq':defect_freq,
                    'recent':inspection_log[-20:],'total_cost':total_cost,
                    'pass_rate':round(100*verdicts['PASS']/max(len(inspection_log),1),1)})

@app.route('/api/shift-report')
def shift_report():
    last_n=int(request.args.get('n',50))
    log_slice=inspection_log[-last_n:]
    return jsonify({'narrative':generate_shift_narrative(log_slice),'count':len(log_slice),
                    'total_cost':sum(r.get('cost_inr',0) for r in log_slice),'log':log_slice})

@app.route('/api/grades')
def grades(): return jsonify(list(GRADE_COST.keys()))

@app.route('/api/seed-demo')
def seed_demo():
    for i in range(15):
        cls_list=random.sample(list(DEFECTS.keys()),random.randint(0,3))
        sevs=[DEFECTS[c]['severity'] for c in cls_list]
        max_sev='High' if 'High' in sevs else ('Medium' if 'Medium' in sevs else ('Low' if sevs else 'None'))
        verdict='PASS' if not cls_list else ('FAIL' if max_sev=='High' else 'REVIEW')
        dets=[{'class':c,'confidence':round(random.uniform(0.72,0.97),2),'area_pct':round(random.uniform(2,15),1)} for c in cls_list]
        grade=random.choice(list(GRADE_COST.keys()))
        cost=calc_cost(dets,grade)
        ago=datetime.datetime.now()-datetime.timedelta(hours=random.randint(0,8),minutes=random.randint(0,59))
        inspection_log.append({'id':str(uuid.uuid4())[:8],'coil_id':f'COIL-{random.randint(10000,99999)}',
                               'timestamp':ago.strftime('%Y-%m-%d %H:%M:%S'),'filename':f'demo_{i}.jpg',
                               'defect_count':len(cls_list),'verdict':verdict,'avg_confidence':round(random.uniform(0.75,0.96),2),
                               'max_severity':max_sev,'grade':grade,'cost_inr':cost['total'],'defects':cls_list})
    return jsonify({'seeded':15})

if __name__=='__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)
    app.run(debug=True,port=5000)
