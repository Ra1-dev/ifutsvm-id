"""Course canonical slide-size vector/PNG figures and editable diagrams."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import textwrap
from xml.etree import ElementTree as ET
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Rectangle
from matplotlib.path import Path as MplPath
from experiments.display import MODEL_NAMES as LABELS,model_name

COLORS={'svm_balanced':'#0072B2','smote_svm':'#E69F00','tsvm':'#56B4E9','utsvm':'#CC79A7',
    'faithful_raw':'#666666','faithful_scaled':'#999999','paper':'#000000',
    'random_average':'#000000','conditional_gaussian':'#009E73'}
DPI=320
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':17,
    'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.bbox':None})


def fitted_slide_font_points(font_points, canvas_pixels, dpi):
    """Equivalent type size after fitting the whole image inside 1280x720."""
    width,height=canvas_pixels
    return font_points*min(1280/width,720/height)*dpi/100


def subplot_label_check(fig):
    """Adjacent subplot titles and axis labels must not collide."""
    renderer=fig.canvas.get_renderer()
    labels=[(index,text) for index,ax in enumerate(fig.axes)
            for text in [ax.title,ax.xaxis.label,ax.yaxis.label]
            if text.get_visible() and text.get_text().strip()]
    checked=0
    for index,(axis,first) in enumerate(labels):
        for other_axis,second in labels[index+1:]:
            if axis==other_axis:continue
            checked+=1
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)), (
                'subplot label overlap',first.get_text(),second.get_text())
    return checked


def saved_png_text_bounds(fig,path):
    """Measure drawn artists at the shipped PNG's DPI and pixel dimensions."""
    from PIL import Image
    with Image.open(path) as png:
        size=png.size;dpi=png.info['dpi']
    fig.canvas.draw()
    assert tuple(fig.canvas.get_width_height())==size,'Renderer and saved PNG canvas differ'
    assert all(abs(value-fig.dpi)<.1 for value in dpi),'Renderer and saved PNG DPI differ'
    omitted=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in [*axis.majorTicks,*axis.minorTicks]:
                if not lo<=tick.get_loc()<=hi:omitted.update([tick.label1,tick.label2])
    texts=[t for t in fig.findobj(match=matplotlib.text.Text)
        if t not in omitted and t.get_visible() and t.get_text().strip()]
    margin=8*size[0]/1280
    rows=[]
    for text in texts:
        box=text.get_window_extent(fig.canvas.get_renderer())
        assert box.x0>=margin and box.x1<=size[0]-margin and box.y0>=margin and box.y1<=size[1]-margin, (path.name,text.get_text(),'saved PNG text margin',box.bounds,size)
        rows.append(dict(text=text.get_text(),bounds_pixels=[box.x0,box.y0,box.x1,box.y1],font_points=text.get_fontsize()))
    return dict(canvas_pixels=list(size),render_dpi=fig.dpi,minimum_margin_at_1280_pixels=8,texts=rows)


class Pack:
    def __init__(self,summary,raw,output,preview):
        self.s=summary;self.raw=raw;self.output=Path(output);self.preview=preview;self.records=[];self.pending=[]
        self.output.mkdir(parents=True,exist_ok=True)

    def save(self,fig,name,title,caption,pointers,editable=None):
        height_scale=7.2/fig.get_figheight()
        fig.suptitle(title,x=.045,y=1-.025*height_scale,ha='left',fontsize=24,fontweight='bold',color='#172F54')
        # Leave margin for font-metric differences between Agg and PDF backends.
        wrapped='\n'.join(textwrap.wrap(caption,width=105))
        fig.text(.045,1-.085*height_scale,wrapped,va='top',fontsize=14,color='#3B4A59')
        fig.text(.045,.023*height_scale,'DEVELOPMENT PREVIEW | Research/final packaging pending' if self.preview else
            'CSCI447P031 | Saved source: results/summary.json',fontsize=14,color='#945100' if self.preview else '#3B4A59')
        # The renderer must use the exact saved DPI: lower-DPI text metrics
        # previously missed a clipped subtitle in the delivered image.
        fig.set_dpi(DPI)
        fig.canvas.draw()
        label_pairs=subplot_label_check(fig)
        # Locators can create ticks outside the visible interval; Matplotlib
        # omits those artists at draw time even though their visible flag is set.
        omitted=set()
        for ax in fig.axes:
            for axis in (ax.xaxis,ax.yaxis):
                lo,hi=sorted(axis.get_view_interval())
                for tick in [*axis.majorTicks,*axis.minorTicks]:
                    if not lo<=tick.get_loc()<=hi:omitted.update([tick.label1,tick.label2])
        texts=[t for t in fig.findobj(match=matplotlib.text.Text) if t not in omitted and t.get_visible() and t.get_text().strip()]
        minimum_font=min(t.get_fontsize() for t in texts)
        assert minimum_font>=14, (name,minimum_font)
        fitted_font=fitted_slide_font_points(minimum_font,fig.canvas.get_width_height(),fig.dpi)
        assert fitted_font>=14-1e-8,(name,'font too small when fitted to slide',fitted_font)
        canvas=fig.bbox
        for text in texts:
            box=text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0>=canvas.x0-1 and box.x1<=canvas.x1+1 and box.y0>=canvas.y0-1 and box.y1<=canvas.y1+1, (name,text.get_text(),'outside figure')
        files=[]
        for extension in ('png','pdf'):
            path=self.output/(name+'.'+extension)
            metadata={'Software':'Matplotlib'} if extension=='png' else {'Creator':'Matplotlib','Producer':'Matplotlib','CreationDate':None}
            fig.savefig(path,dpi=DPI,facecolor='white',metadata=metadata)
            files.append(dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if extension=='png':png_bounds=saved_png_text_bounds(fig,path)
        # Inspect exported vector text, not just the pre-export Agg canvas.
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(self.output/(name+'.pdf')) as document:
            page=document[0];textpage=page.get_textpage();pdf_characters=textpage.count_chars()
            for index in range(pdf_characters):
                left,bottom,right,top=textpage.get_charbox(index)
                assert left>=-.5 and bottom>=-.5 and right<=page.get_width()+.5 and top<=page.get_height()+.5, (name,index,'PDF text outside page')
        plt.close(fig)
        if editable:files.append(dict(path=editable.as_posix(),sha256=hashlib.sha256(editable.read_bytes()).hexdigest()))
        self.records.append(dict(name=name,title=title,caption=caption,source_pointers=pointers,files=files,
            measured_minimum_font_points=minimum_font,text_bounds_checked=len(texts),
            minimum_fitted_slide_font_points=fitted_font,slide_canvas_pixels=[1280,720],
            pdf_character_bounds_checked=pdf_characters,
            saved_png_text_bounds=png_bounds,
            subplot_label_pairs_checked=label_pairs,
            command='python -m experiments.presentation_figures'+(' --preview' if self.preview else '')))

    def diagram(self,name,title,caption,nodes,edges,pointers,horizontal=False):
        compact=name=='fig01_methodology_block_diagram'
        fig=plt.figure(figsize=(12.8,5.2 if compact else 7.2))
        ax=fig.add_axes([.035,.10 if compact else .09,.93,.64 if compact else .69])
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        for label,(x,y,w,h,text) in nodes.items():
            ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.008',
                facecolor='#EEF4F7',edgecolor='#55748B',linewidth=1.4))
            ax.text(x,y,text,ha='center',va='center',fontsize=14,linespacing=1.3)
        for edge in edges:
            first,second=edge[:2]
            a=nodes[first];b=nodes[second];dx=b[0]-a[0];dy=b[1]-a[1]
            if len(edge)>2:
                side=-1 if edge[2]>0 else 1
                start=(a[0]+side*a[2]/2,a[1]);end=(b[0]+side*b[2]/2,b[1])
                route=a[0]+side*(a[2]/2+.04)
                path=MplPath([start,(route,a[1]),(route,b[1]),end],
                    [MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
                ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=17,color='#2B706B',linewidth=1.6))
                continue
            if horizontal or abs(dx)>abs(dy):start=(a[0]+np.sign(dx)*a[2]/2,a[1]);end=(b[0]-np.sign(dx)*b[2]/2,b[1])
            else:start=(a[0],a[1]+np.sign(dy)*a[3]/2);end=(b[0],b[1]-np.sign(dy)*b[3]/2)
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=17,color='#2B706B',
                linewidth=1.6,shrinkA=6,shrinkB=6,connectionstyle=f'arc3,rad={edge[2] if len(edge)>2 else 0}'))
        mxfile=ET.Element('mxfile',host='app.diagrams.net');diagram=ET.SubElement(mxfile,'diagram',id=name,name=title)
        graph=ET.SubElement(diagram,'mxGraphModel',dx='1280',dy='720',grid='1',page='1',pageWidth='1280',pageHeight='720')
        graphroot=ET.SubElement(graph,'root');ET.SubElement(graphroot,'mxCell',id='0');ET.SubElement(graphroot,'mxCell',id='1',parent='0')
        for label,(x,y,w,h,text) in nodes.items():
            cell=ET.SubElement(graphroot,'mxCell',id=label,value=text.replace('\n','<br>'),parent='1',vertex='1',
                style='rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF4F7;strokeColor=#55748B;fontSize=20;')
            ET.SubElement(cell,'mxGeometry',{'x':str(40+1200*(x-w/2)),'y':str(120+500*(1-y-h/2)),
                'width':str(1200*w),'height':str(500*h),'as':'geometry'})
        for i,edge in enumerate(edges):
            first,second=edge[:2]
            style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor=#2B706B;'
            if horizontal:style='edgeStyle=none;html=1;endArrow=block;strokeColor=#2B706B;exitX=1;exitY=0.5;entryX=0;entryY=0.5;'
            if len(edge)>2:
                anchor='0' if edge[2]>0 else '1';style+=f'exitX={anchor};exitY=0.5;entryX={anchor};entryY=0.5;'
            cell=ET.SubElement(graphroot,'mxCell',id=f'edge-{i}',parent='1',edge='1',source=first,target=second,style=style)
            geometry=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
            if len(edge)>2:
                a=nodes[first];b=nodes[second];x=a[0]+(-1 if edge[2]>0 else 1)*(a[2]/2+.04)
                waypoints=ET.SubElement(geometry,'Array',{'as':'points'})
                for y in (a[1],b[1]):ET.SubElement(waypoints,'mxPoint',x=str(40+1200*x),y=str(120+500*(1-y)))
        path=self.output/'src'/(name+'.drawio');path.parent.mkdir(exist_ok=True)
        ET.indent(mxfile);path.write_text(ET.tostring(mxfile,encoding='unicode')+'\n',encoding='utf-8',newline='\n')
        self.save(fig,name,title,caption,pointers,editable=path)

    def diagrams(self):
        self.diagram('fig01_methodology_block_diagram','Methodology: train, then evaluate',
            'Scaling, Universum construction and fuzzy scores use training rows only; inner validation chooses parameters.',{
            'data':(.12,.76,.21,.26,'Official KEEL data\nMinority = positive'),
            'split':(.37,.76,.21,.26,'Stratified split\nTraining / test'),
            'scale':(.63,.76,.21,.26,'Train-only\nencoding and scaling'),
            'u':(.88,.76,.21,.26,'Training Universum\nU and subset U*'),
            'metrics':(.12,.28,.21,.26,'Metrics and statistics\nPersist predictions'),
            'decision':(.37,.28,.21,.26,'Decision rule\nNearest plane'),
            'qp':(.63,.28,.21,.26,'Two convex QPs\nFit both planes'),
            'if':(.88,.28,.21,.26,'IF scores\nTraining geometry')},
            [('data','split'),('split','scale'),('scale','u'),('u','if'),('if','qp'),('qp','decision'),('decision','metrics')],['/config'])
        self.diagram('fig02_model_architecture',model_name('paper')+': two fitted planes',
            'Reduced kernel basis D = [X1; X2*]; default symmetric Universum constraints; all kernel blocks share bandwidth.',{
            'classes':(.085,.50,.15,.24,'Minority X1\nMajority X2'),
            'prepare':(.285,.50,.18,.40,'Draw X2*, U, U*\nKernel IF scores\ns_min, s_maj*'),
            'kernel':(.50,.50,.19,.32,'Kernel blocks\nE, F*, G*, F2, G\nD = [X1; X2*]'),
            'q1':(.715,.76,.17,.24,'Plane-1 QP\nMinority plane z1'),
            'q2':(.715,.24,.17,.24,'Plane-2 QP\nMajority plane z2'),
            'decision':(.91,.50,.15,.32,'Equation 32\nMinimum\nplane distance')},
            [('classes','prepare'),('prepare','kernel'),('kernel','q1'),('kernel','q2'),
             ('q1','decision'),('q2','decision')],['/config'],horizontal=True)
        c=self.s['config']
        self.diagram('fig03_evaluation_protocol','Two outer evaluation protocols',
            'Outer test data never enter tuning, scaling, SMOTE or Universum fitting. Main selection: balanced accuracy.',{
            'a':(.25,.82,.43,.19,'Protocol A: 70/30 holdout\nSeeds 0, 1, 2, 3, 4'),
            'b':(.75,.82,.43,.19,'Protocol B: nested 5-fold CV\nOuter shuffle seed 0'),
            'at':(.25,.50,.43,.19,f"Training only: {c['inner_folds']}-fold inner CV\n{c['budget']} shared candidate proposals"),
            'bt':(.75,.50,.43,.19,f"Training only: {c['inner_folds']}-fold inner CV\n{c['budget']} shared candidate proposals"),
            'ae':(.25,.18,.43,.19,'Refit on the full outer training set\nEvaluate once on the held-out 30%'),
            'be':(.75,.18,.43,.19,'Refit on the full outer training set\nEvaluate once on the held-out fold')},
            [('a','at'),('at','ae'),('b','bt'),('bt','be')],['/config'])

    def performance(self):
        names=self.s['config']['models'];fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),sharey=True)
        fig.subplots_adjust(left=.25,right=.96,top=.75,bottom=.16,wspace=.17)
        for ax,metric,title in zip(axes,['balanced_accuracy','gmean'],['Balanced accuracy','G-mean']):
            for protocol,offset,marker in [('holdout',-.14,'o'),('nested_cv',.14,'s')]:
                values=self.s['protocols'][protocol]['aggregate']['models']
                for i,name in enumerate(names):
                    row=values[name][metric]
                    ax.errorbar(100*row['mean'],i+offset,xerr=100*row['std'],fmt=marker,
                        color=COLORS[name],capsize=4,markersize=6,linewidth=1.5)
            ax.set(title=title,xlim=(0,100),xlabel='Score (%)',xticks=[0,25,50,75,100]);ax.grid(axis='x',alpha=.2)
        axes[0].set_yticks(range(len(names)),[model_name(n,width=25) for n in names]);axes[0].invert_yaxis()
        axes[1].legend(handles=[Line2D([],[],color='#333333',marker=m,linestyle='',label=l)
            for m,l in [('o','Holdout'),('s','Outer CV')]],loc='lower left',frameon=False)
        h=self.s['headline'];v=h['balanced_accuracy_percent']
        self.save(fig,'fig04_main_performance',model_name('paper')+' trails simple baselines',
            f"Holdout balanced accuracy: {model_name('paper')} {v['paper']:.1f}%, {model_name('svm_balanced')} {v['svm_balanced']:.1f}%, {model_name('smote_svm')} {v['smote_svm']:.1f}%. "
            f"Equal-weight means over {h['datasets']} tasks; bars show SD between dataset means.",
            ['/headline','/protocols/holdout/aggregate','/protocols/nested_cv/aggregate'])

    def roc(self):
        data=self.s['presentation_illustrations'];names=self.s['config']['models']
        fig,axes=plt.subplots(1,3,figsize=(12.8,7.2));fig.subplots_adjust(left=.07,right=.98,bottom=.29,top=.73,wspace=.26)
        for ax,(dataset,group) in zip(axes,data['datasets'].items()):
            for name in names:
                r=group['models'][name]['roc']
                ax.plot(r['fpr'],r['tpr'],color=COLORS[name],linewidth=2 if name=='paper' else 1.5,
                    linestyle='--' if name=='faithful_scaled' else '-',label=model_name(name,width=32))
            ax.plot([0,1],[0,1],color='#BBBBBB',linestyle=':',linewidth=1)
            ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='False-positive rate',
                title=f"{dataset}\nIR = {group['imbalance_ratio']:.2f}");ax.grid(alpha=.15)
        axes[0].set_ylabel('Minority true-positive rate')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',
            bbox_to_anchor=(.51,.055),frameon=False,columnspacing=1.8,handlelength=2)
        self.save(fig,'fig05_roc_curves','ROC on predeclared imbalance levels',
            'Minimum, lower-median and maximum dataset IR; seed zero. Curves use persisted minority-positive decision scores, with no performance-based dataset choice.',
            ['/artifact_plan/representative_datasets','/presentation_illustrations'])

    def confusion(self):
        data=self.s['presentation_illustrations'];names=data['confusion_models']
        fig,axes=plt.subplots(2,3,figsize=(12.8,7.2));fig.subplots_adjust(left=.115,right=.98,bottom=.13,top=.76,hspace=.90,wspace=.46)
        for column,(dataset,group) in enumerate(data['datasets'].items()):
            for row,name in enumerate(names):
                ax=axes[row,column]
                cm=np.array(group['models'][name]['summed_confusion_matrix']);rates=cm/cm.sum(1,keepdims=True)
                for i in range(2):
                    for j in range(2):
                        ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=plt.cm.Blues(rates[i,j]),edgecolor='white',linewidth=2))
                        ax.text(j,i,f'{cm[i,j]}\n{100*rates[i,j]:.1f}%',ha='center',va='center',fontsize=16,
                            color='white' if rates[i,j]>.55 else '#172F54')
                ax.set(xlim=(-.5,1.5),ylim=(1.5,-.5),xticks=[0,1],yticks=[0,1],
                    xticklabels=['Majority','Minority'],yticklabels=['Majority','Minority'],xlabel='Predicted' if row==1 else '',ylabel='True',
                    title=f'{dataset}\n{LABELS[name]}');ax.set_aspect('auto')
        self.save(fig,'fig06_confusion_matrices','Class errors across all holdout seeds',
            f"{model_name('paper')} versus {LABELS[data['best_baseline']]} (best complete-benchmark mean balanced accuracy). "
            f"Counts summed over {len(data['confusion_seeds'])} seeds; cell shade and percent are row recall (0-100%).",
            ['/presentation_illustrations','/protocols/holdout/aggregate'])

    def curves(self):
        data=self.s.get('course_curves')
        if not data or not data['complete']:self.pending.append('course_curves');return
        def bounds(ax,rows):
            lows=[];highs=[]
            for mean,sd in rows:lows.extend(mean-sd);highs.extend(mean+sd)
            lower=min(0,10*np.floor(min(lows)/10));upper=max(100,10*np.ceil(max(highs)/10))
            ax.set_ylim(lower-2,upper+2);ax.set_yticks(np.arange(lower,upper+1,20))
            ax.grid(alpha=.18)
        fig,axes=plt.subplots(1,2,figsize=(12.8,7.2));fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.75,wspace=.25)
        for ax,metric,title in zip(axes,['accuracy','balanced_accuracy'],['Accuracy','Balanced accuracy']):
            intervals=[]
            for model in data['plan']['learning']['models']:
                rows=data['groups'][model]['learning'];x=[r['training_rows_mean'] for r in rows]
                for role,line,marker in [('training','--','x'),('validation','-','o')]:
                    mean=np.array([r['metrics'][metric][role]['mean'] for r in rows])*100
                    sd=np.array([r['metrics'][metric][role]['std'] for r in rows])*100;intervals.append((mean,sd))
                    ax.errorbar(x,mean,yerr=sd,color=COLORS[model],linestyle=line,marker=marker,capsize=3,
                        linewidth=1.5,label=LABELS[model]+' / '+role)
            bounds(ax,intervals);ax.set(xlim=(0,max(x)*1.06),title=title,xlabel='Training rows (fold mean)',ylabel='Score (%)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.52,.065),frameon=False)
        plan=data['plan']['learning']
        self.save(fig,'fig07_learning_curves','Learning curves: '+model_name('paper')+' and '+model_name('svm_balanced'),
            f"{plan['dataset']}; parameters fixed at original training-selected values. "
            f"Mean and SD over {plan['inner_folds']} inner folds; outer test rows excluded. Descriptive curves, not new tuning.",
            ['/course_curves/plan/learning','/course_curves/groups'])
        fig,axes=plt.subplots(1,2,figsize=(12.8,7.2));fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.75,wspace=.25)
        for ax,family in zip(axes,['C','mu']):
            rows=data['groups']['paper'][family];x=[r['value'] for r in rows];intervals=[]
            for role,line,marker in [('training','--','x'),('validation','-','o')]:
                mean=np.array([r['metrics']['balanced_accuracy'][role]['mean'] for r in rows])*100
                sd=np.array([r['metrics']['balanced_accuracy'][role]['std'] for r in rows])*100;intervals.append((mean,sd))
                ax.errorbar(x,mean,yerr=sd,color=COLORS['paper'],linestyle=line,marker=marker,capsize=4,label=role.title())
            bounds(ax,intervals);ax.set_xscale('log',base=10 if family=='C' else 2)
            ax.set(xlabel='C' if family=='C' else 'Kernel bandwidth (mu)',ylabel='Balanced accuracy (%)');ax.legend(frameon=False,loc='lower right')
        self.save(fig,'fig08_validation_curves','Validation sensitivity within outer training',
            f"{model_name('paper')}, {plan['dataset']}; vary one parameter, with all others fixed at original training-selected values. "
            'Mean and SD across inner folds; descriptive reuse of inner selection, not an unbiased test estimate.',
            ['/course_curves/plan/validation','/course_curves/groups/paper'])

    def reproduction(self):
        rows=self.s['reproduction']['datasets'];names=list(rows)
        fig=plt.figure(figsize=(12.8,7.2));ax=fig.add_axes([.17,.18,.47,.57]);notes=fig.add_axes([.68,.18,.30,.57])
        categories={'mean_within_tolerance':'Within tolerance',
            'configuration_contrast_leaves_residual_within_tolerance':'Configuration contrast',
            'metric_definition_contrast_leaves_residual_within_tolerance':'Balanced-accuracy hypothesis',
            'published_configuration_all_majority_published_below_majority_and_equals_IF_USVM':'Majority collapse; repeated column',
            'published_accuracy_unattained_by_preregistered_four_family_probe':'Pre-specified accuracy probe',
            'residual_unresolved':'Unresolved'}
        for key,offset,marker,color,label in [
            ('published_accuracy_percent',-.18,'D','#888888','Published ACC'),
            ('tuned_accuracy_mean_percent',0,'o',COLORS['paper'],'Our accuracy'),
            ('tuned_balanced_accuracy_mean_percent',.18,'s',COLORS['paper'],'Our balanced accuracy')]:
            ax.scatter([rows[n]['comparison'][key] for n in names],np.arange(len(names))+offset,
                marker=marker,s=32,edgecolor=color,facecolor='white' if marker=='s' else color,label=label,zorder=3)
        for i,name in enumerate(names):notes.text(0,i,categories[rows[name]['comparison']['evidence_category']],va='center',fontsize=14)
        ax.set(yticks=range(len(names)),yticklabels=names,xlim=(0,100),ylim=(len(names)-.6,-.6),xlabel='Score (%)')
        ax.grid(axis='x',alpha=.18);notes.set(xlim=(0,1),ylim=(len(names)-.6,-.6));notes.axis('off')
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False,columnspacing=1)
        self.save(fig,'fig09_reproduction','Published accuracies: matches and measured gaps',
            f"{model_name('paper')}: accuracy-selected holdout means; balanced accuracy uses the same predictions. Explanation categories do not identify undocumented author settings.",
            ['/reproduction/datasets'])

    def critical_difference(self):
        data=self.s['main_statistics']['friedman_nemenyi'];fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),sharey=True)
        fig.subplots_adjust(left=.34,right=.97,bottom=.16,top=.74,wspace=.22)
        ordered=sorted(data['holdout']['balanced_accuracy']['models'],
            key=lambda n:data['holdout']['balanced_accuracy']['average_ranks'][n])
        for ax,protocol,title in zip(axes,['holdout','nested_cv'],['Holdout','Outer CV']):
            r=data[protocol]['balanced_accuracy'];names=r['models']
            for y,name in enumerate(ordered):ax.scatter(r['average_ranks'][name],y,color=COLORS[name],s=60,zorder=3)
            ax.set(yticks=range(len(names)),yticklabels=[model_name(n) for n in ordered],ylim=(len(names)-.5,-1.5),
                xlim=(.8,len(names)+.2),xticks=range(1,len(names)+1),xlabel='Mean rank (lower is better)',
                title=f"{title}: p = {r['friedman_p']:.3g}\n{r['datasets']} datasets")
            cd=r['critical_difference'];ax.annotate('',xy=(1+cd,-.65),xytext=(1,-.65),
                arrowprops=dict(arrowstyle='|-|',color='#009E73',linewidth=1.6))
            ax.text(1+cd/2,-.95,f'CD = {cd:.3f}',ha='center',fontsize=14);ax.grid(axis='x',alpha=.18)
        self.save(fig,'fig10_critical_difference','Balanced-accuracy critical differences',
            'Friedman/Nemenyi across all tasks; each predeclared protocol is a separate test family. The ruler shows the critical rank difference; post-hoc inference uses the omnibus gate.',
            ['/main_statistics/friedman_nemenyi'])

    def ablation(self):
        research=self.s.get('research')
        if not research or not research['complete']:self.pending.append('ablation');return
        names=list(research['plan']['variants']);datasets=research['plan']['datasets']
        matrix=np.array([[research['ablation']['datasets'][d]['0.0'][n]['metrics']['gmean']['paired_vs_random']['mean']
            for n in names] for d in datasets])*100;means=matrix.mean(0);sd=matrix.std(0,ddof=1)
        fig,ax=plt.subplots(figsize=(12.8,7.2));fig.subplots_adjust(left=.53,right=.97,bottom=.16,top=.79)
        for i,name in enumerate(names):
            color=COLORS['conditional_gaussian'] if name.startswith('conditional') else '#000000' if name.startswith('random') else '#666666'
            ax.errorbar(means[i],i,xerr=sd[i],fmt='o',color=color,capsize=4,markersize=6)
        ax.axvline(0,color='#AAAAAA',linewidth=1);ax.set(yticks=range(len(names)),yticklabels=[model_name(n) for n in names],
            xlabel='Paired G-mean difference (percentage points)');ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        self.save(fig,'fig11_ablation','All pre-specified Universum contrasts',
            f"Paired differences versus {model_name('random_average')}: equal-weight mean and SD across {len(datasets)} dataset means. Every pre-specified variant is retained.",
            ['/research/ablation','/research/plan/variants'])

    def noise(self):
        research=self.s.get('research')
        if not research or not research['complete']:self.pending.append('noise');return
        plan=research['plan'];levels=plan['noise_levels'];names=plan['noise_models'];datasets=plan['noise_datasets']
        intervals=[research['noise']['datasets'][d][str(float(level))][name]['metrics']['balanced_accuracy']
            for d in datasets for level in levels for name in names]
        lower=min(0,10*np.floor(min(100*(r['mean']-r['std']) for r in intervals)/10))-2
        upper=max(100,10*np.ceil(max(100*(r['mean']+r['std']) for r in intervals)/10))+2
        fig,axes=plt.subplots(2,3,figsize=(12.8,7.2));fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.74,wspace=.35,hspace=.62)
        for ax,dataset in zip(axes.flat,datasets):
            for name in names:
                rows=[research['noise']['datasets'][dataset][str(float(level))][name]['metrics']['balanced_accuracy'] for level in levels]
                ax.errorbar(np.array(levels)*100,[100*r['mean'] for r in rows],yerr=[100*r['std'] for r in rows],
                    color=COLORS[name],marker='o',capsize=3,label=model_name(name,width=25),linewidth=1.6)
            ax.set(title=dataset,
                xticks=np.array(levels)*100,ylim=(lower,upper),yticks=[0,25,50,75,100]);ax.grid(alpha=.18)
        fig.text(.025,.46,'Balanced accuracy (%)',va='center',rotation=90,fontsize=15)
        fig.text(.52,.075,'Training labels flipped (%)',ha='center',fontsize=15)
        legend=axes.flat[-1];legend.axis('off');handles,text=axes.flat[0].get_legend_handles_labels()
        legend.legend(handles,text,loc='center',frameon=False)
        self.save(fig,'fig12_label_noise','Robustness to corrupted training labels',
            'Only training labels are flipped; outer test labels remain clean. Identical flip permutations and tuning budgets for all models. Mean and SD across the fixed holdout seeds.',
            ['/research/noise','/research/plan/noise_levels','/research/plan/noise_models'])



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--output')
    parser.add_argument('--verification-output')
    args=parser.parse_args();raw=Path('results/summary.json').read_bytes();summary=json.loads(raw)
    output=Path(args.output or ('tmp/presentation-preview' if args.preview else 'figures'))
    if args.preview and output.resolve()==Path('figures').resolve():raise ValueError('Preview cannot overwrite final figures')
    methods=('diagrams','performance','roc','confusion','curves','reproduction','critical_difference','ablation','noise')
    if not args.preview:
        assert all(hasattr(Pack,name) for name in methods),'Canonical figure implementation is incomplete'
        assert summary['complete'] and summary.get('research',{}).get('complete') and summary.get('course_curves',{}).get('complete')
        from experiments.snapshot import require_snapshot
        require_snapshot()
    pack=Pack(summary,raw,output,args.preview)
    for name in methods:
        if hasattr(pack,name):getattr(pack,name)()
        else:pack.pending.append(name)
    from PIL import Image
    for record in pack.records:
        with Image.open(output/(record['name']+'.png')) as png:
            assert min(png.info['dpi'])>=300
            record['png_pixels']=png.size;record['measured_dpi']=png.info['dpi']
    if not args.preview:assert len(pack.records)==12 and not pack.pending
    manifest=dict(preview=args.preview,summary_sha256=hashlib.sha256(raw).hexdigest(),figures=pack.records,pending=pack.pending,
        minimum_font_points=14,reference_width_pixels=1280,model_colors=COLORS,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.verification_output:
        Path(args.verification_output).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    captions=['# Figure captions and sources','']
    for r in pack.records:
        captions += ['## '+r['name'],'',r['caption'],'','- Source: `results/summary.json` '+', '.join('`'+p+'`' for p in r['source_pointers']),
            '- Command: `'+r['command']+'`.','']
    (output/'CAPTIONS.md').write_text('\n'.join(captions),encoding='utf-8',newline='\n')
    print(f'Presentation figures: {len(pack.records)}/12; pending={pack.pending}; preview={args.preview}')


if __name__=='__main__':main()
