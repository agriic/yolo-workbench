const state={meta:null,images:[],current:null,image:null,selected:null,drawing:null,scale:1,offset:{x:0,y:0},drag:null};
const $=id=>document.getElementById(id);
const api=async(path,options={})=>{const response=await fetch(path,{headers:{"Content-Type":"application/json"},...options});if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(body.detail||response.statusText)}return response.json()};
const colors=["#00b894","#e17055","#0984e3","#fdcb6e","#e84393","#6c5ce7","#d63031","#2d3436"];
const toast=message=>{const el=$("toast");el.textContent=message;el.style.display="block";clearTimeout(el.timer);el.timer=setTimeout(()=>el.style.display="none",3000)};
const esc=value=>String(value).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));

async function init(){
  state.meta=await api("/api/v1/dataset");
  $("dataset-name").textContent=`${state.meta.yaml} · ${state.meta.category}`;
  $("issue-count").textContent=state.meta.issue_count?`(${state.meta.issue_count})`:"";
  for(const select of [$("image-class"),$("object-class"),$("draw-class")]) for(const [id,name] of Object.entries(state.meta.names)) select.insertAdjacentHTML("beforeend",`<option value="${id}">${esc(id)} · ${esc(name)}</option>`);
  for(const select of [$("image-split"),$("object-split")]) for(const split of Object.keys(state.meta.split_counts)) select.insertAdjacentHTML("beforeend",`<option>${esc(split)}</option>`);
  bind();await Promise.all([loadImages(),loadObjects(),loadIssues()]);
}

function bind(){
  document.querySelectorAll(".tab").forEach(button=>button.onclick=()=>{document.querySelectorAll(".tab,.view").forEach(el=>el.classList.remove("active"));button.classList.add("active");$(button.dataset.tab).classList.add("active")});
  [$("image-split"),$("image-class")].forEach(el=>el.onchange=loadImages);$("image-search").oninput=debounce(loadImages,250);
  [$("object-class"),$("object-split"),$("crop-padding")].forEach(el=>el.oninput=loadObjects);
  $("refresh-issues").onclick=loadIssues;$("close-editor").onclick=()=>$("editor").close();$("fit").onclick=fit;$("zoom-in").onclick=()=>zoom(1.25);$("zoom-out").onclick=()=>zoom(.8);
  $("undo").onclick=()=>history("undo");$("redo").onclick=()=>history("redo");$("delete").onclick=deleteSelected;
  const canvas=$("canvas");canvas.onpointerdown=pointerDown;canvas.onpointermove=pointerMove;canvas.onpointerup=pointerUp;canvas.onpointerleave=pointerUp;
  window.onkeydown=event=>{if(!$("editor").open)return;if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();history(event.shiftKey?"redo":"undo")}else if(event.key==="Delete"||event.key==="Backspace")deleteSelected();else if(event.key==="Escape")state.drawing=null,draw()};
}

async function loadImages(){
  const params=new URLSearchParams({split:$("image-split").value,search:$("image-search").value,limit:"500"});if($("image-class").value!=="")params.set("class_id",$("image-class").value);
  const data=await api(`/api/v1/images?${params}`);state.images=data.items;$("image-total").textContent=`${data.total} images`;
  $("image-grid").innerHTML=data.items.map(item=>`<article class="card" data-image="${item.id}" tabindex="0"><img loading="lazy" src="/api/v1/images/${item.id}/thumbnail"><div class="card-body"><div class="name" title="${esc(item.name)}">${esc(item.name)}</div><div class="meta"><span>${esc(item.split)} · ${item.annotation_count} objects</span><span class="${item.issue_count?'bad':''}">${item.issue_count?item.issue_count+' issues':''}</span></div></div></article>`).join("")||`<p class="muted">No matching images.</p>`;
  document.querySelectorAll("[data-image]").forEach(card=>{card.onclick=()=>openEditor(card.dataset.image);card.onkeydown=e=>{if(e.key==="Enter")openEditor(card.dataset.image)}});
}

async function openEditor(id,selected=null){
  state.current=id;state.currentSelected=selected;state.current=await api(`/api/v1/images/${id}`);state.selected=selected;state.drawing=null;
  state.image=new Image();state.image.onload=()=>{fit();renderList()};state.image.src=`/api/v1/images/${id}/file?${Date.now()}`;
  $("editor-title").textContent=state.current.name;$("editor-help").textContent=state.meta.category==="detection"?"Drag empty space to create a box. Drag a box to move it; use corner handles to resize.":"Click empty space to add polygon vertices; click the first point to close. Drag vertices to reshape.";
  $("editor").showModal();
}

function fit(){if(!state.image)return;const wrap=$("canvas-wrap"),canvas=$("canvas"),ratio=Math.min((wrap.clientWidth-20)/state.image.width,(wrap.clientHeight-20)/state.image.height);state.scale=Math.max(.05,ratio);canvas.width=Math.floor(state.image.width*state.scale);canvas.height=Math.floor(state.image.height*state.scale);draw()}
function zoom(factor){if(!state.image)return;state.scale=Math.min(8,Math.max(.05,state.scale*factor));const canvas=$("canvas");canvas.width=Math.floor(state.image.width*state.scale);canvas.height=Math.floor(state.image.height*state.scale);draw()}
const toCanvas=(x,y)=>[x*$("canvas").width,y*$("canvas").height];const toNorm=e=>{const rect=$("canvas").getBoundingClientRect();return [Math.min(1,Math.max(0,(e.clientX-rect.left)/rect.width)),Math.min(1,Math.max(0,(e.clientY-rect.top)/rect.height))]};

function draw(){
  const canvas=$("canvas"),ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);if(!state.image)return;ctx.drawImage(state.image,0,0,canvas.width,canvas.height);
  state.current.annotations.forEach(annotation=>drawAnnotation(ctx,annotation,annotation.id===state.selected));
  if(state.drawing){ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.setLineDash([6,4]);if(state.meta.category==="detection"){const [x1,y1]=toCanvas(state.drawing.start[0],state.drawing.start[1]),[x2,y2]=toCanvas(state.drawing.end[0],state.drawing.end[1]);ctx.strokeRect(x1,y1,x2-x1,y2-y1)}else{const pts=state.drawing.points.map(p=>toCanvas(...p));ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke()}ctx.setLineDash([])}
}
function drawAnnotation(ctx,a,selected){ctx.strokeStyle=colors[a.class_id%colors.length];ctx.fillStyle=colors[a.class_id%colors.length];ctx.lineWidth=selected?4:2;if(state.meta.category==="detection"){const [cx,cy,w,h]=a.points,[x,y]=toCanvas(cx-w/2,cy-h/2),[right,bottom]=toCanvas(cx+w/2,cy+h/2);ctx.strokeRect(x,y,right-x,bottom-y);if(selected)for(const p of [[x,y],[right,y],[right,bottom],[x,bottom]])ctx.fillRect(p[0]-5,p[1]-5,10,10)}else{const pts=[];for(let i=0;i<a.points.length;i+=2)pts.push(toCanvas(a.points[i],a.points[i+1]));ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.stroke();if(selected)pts.forEach(p=>{ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.fill()})}}

function hit(x,y){
  for(const a of [...state.current.annotations].reverse()){
    if(state.meta.category==="detection"){const [cx,cy,w,h]=a.points,corners=[[cx-w/2,cy-h/2],[cx+w/2,cy-h/2],[cx+w/2,cy+h/2],[cx-w/2,cy+h/2]];for(let i=0;i<corners.length;i++)if(Math.hypot((corners[i][0]-x)*$("canvas").width,(corners[i][1]-y)*$("canvas").height)<12)return {a,mode:"resize",index:i};if(x>=cx-w/2&&x<=cx+w/2&&y>=cy-h/2&&y<=cy+h/2)return {a,mode:"move"}}
    else{for(let i=0;i<a.points.length;i+=2)if(Math.hypot((a.points[i]-x)*$("canvas").width,(a.points[i+1]-y)*$("canvas").height)<12)return {a,mode:"vertex",index:i};if(pointInPolygon(x,y,a.points))return {a,mode:"move"}}
  }return null;
}
function pointerDown(e){
  const p=toNorm(e),found=hit(...p);$("canvas").setPointerCapture(e.pointerId);
  if(found){state.selected=found.a.id;state.drag={...found,start:p,original:[...found.a.points]};renderList();draw();return}
  if(state.meta.category==="detection"){state.selected=null;state.drawing={start:p,end:p}}
  else{if(!state.drawing)state.drawing={points:[p]};else if(state.drawing.points.length>=3&&Math.hypot((state.drawing.points[0][0]-p[0])*$("canvas").width,(state.drawing.points[0][1]-p[1])*$("canvas").height)<14)finishPolygon();else state.drawing.points.push(p)}draw();
}
function pointerMove(e){const p=toNorm(e);if(state.drawing&&state.meta.category==="detection"){state.drawing.end=p;draw()}if(!state.drag)return;const dx=p[0]-state.drag.start[0],dy=p[1]-state.drag.start[1],a=state.drag.a;if(state.drag.mode==="vertex"){a.points[state.drag.index]=p[0];a.points[state.drag.index+1]=p[1]}else if(state.meta.category==="detection"&&state.drag.mode==="resize"){const [cx,cy,w,h]=state.drag.original,opposites=[[cx+w/2,cy+h/2],[cx-w/2,cy+h/2],[cx-w/2,cy-h/2],[cx+w/2,cy-h/2]],opposite=opposites[state.drag.index];a.points=[(p[0]+opposite[0])/2,(p[1]+opposite[1])/2,Math.max(.001,Math.abs(p[0]-opposite[0])),Math.max(.001,Math.abs(p[1]-opposite[1]))]}else if(state.meta.category==="detection"){a.points[0]=Math.min(1,Math.max(0,state.drag.original[0]+dx));a.points[1]=Math.min(1,Math.max(0,state.drag.original[1]+dy))}else for(let i=0;i<a.points.length;i+=2){a.points[i]=Math.min(1,Math.max(0,state.drag.original[i]+dx));a.points[i+1]=Math.min(1,Math.max(0,state.drag.original[i+1]+dy))}draw()}
async function pointerUp(){if(state.drag){state.drag=null;await save()}else if(state.drawing&&state.meta.category==="detection"){const {start,end}=state.drawing;state.drawing=null;const w=Math.abs(end[0]-start[0]),h=Math.abs(end[1]-start[1]);if(w>.002&&h>.002){state.current.annotations.push({id:null,class_id:+$("draw-class").value,points:[(start[0]+end[0])/2,(start[1]+end[1])/2,w,h]});await save()}draw()}}
async function finishPolygon(){const points=state.drawing.points.flat();state.drawing=null;state.current.annotations.push({id:null,class_id:+$("draw-class").value,points});await save()}

function renderList(){if(!state.current)return;$("annotation-list").innerHTML=state.current.annotations.map((a,index)=>`<div class="annotation-row ${a.id===state.selected?'selected':''}" data-annotation="${esc(a.id)}"><span class="swatch" style="background:${colors[a.class_id%colors.length]}"></span><span>#${index+1}</span><select data-relabel="${esc(a.id)}">${Object.entries(state.meta.names).map(([id,name])=>`<option value="${id}" ${+id===a.class_id?'selected':''}>${esc(name)}</option>`).join("")}</select></div>`).join("");document.querySelectorAll("[data-annotation]").forEach(row=>row.onclick=e=>{if(e.target.tagName!=="SELECT"){state.selected=row.dataset.annotation;renderList();draw()}});document.querySelectorAll("[data-relabel]").forEach(select=>select.onchange=async()=>{state.current.annotations.find(a=>a.id===select.dataset.relabel).class_id=+select.value;await save()})}
async function save(){try{state.current=await api(`/api/v1/images/${state.current.id}/annotations`,{method:"PUT",body:JSON.stringify({annotations:state.current.annotations})});renderList();draw();loadImages();loadIssues()}catch(error){toast(error.message);state.current=await api(`/api/v1/images/${state.current.id}`);renderList();draw()}}
async function deleteSelected(){if(!state.selected)return;state.current.annotations=state.current.annotations.filter(a=>a.id!==state.selected);state.selected=null;await save()}
async function history(direction){try{const result=await api(`/api/v1/history/${direction}`,{method:"POST"});if(state.current&&result.image_id===state.current.id){state.current=await api(`/api/v1/images/${state.current.id}`);renderList();draw()}await Promise.all([loadImages(),loadObjects(),loadIssues()])}catch(error){toast(error.message)}}

async function loadObjects(){if(!state.meta)return;const classId=$("object-class").value||Object.keys(state.meta.names)[0];const params=new URLSearchParams({class_id:classId,split:$("object-split").value,limit:"500"});const data=await api(`/api/v1/objects?${params}`);$("object-total").textContent=`${data.total} objects`;const padding=$("crop-padding").value;$("object-grid").innerHTML=data.items.map(item=>`<article class="card object-card"><img loading="lazy" src="/api/v1/objects/${item.image_id}/${encodeURIComponent(item.id)}/crop?padding=${padding}"><div class="card-body"><div class="name">${esc(item.image_name)}</div><div class="meta"><span>${esc(item.split)}</span></div><div class="object-actions"><select data-object-class="${esc(item.id)}" data-owner="${item.image_id}">${Object.entries(state.meta.names).map(([id,name])=>`<option value="${id}" ${+id===item.class_id?'selected':''}>${esc(name)}</option>`).join("")}</select><button data-open-object="${esc(item.id)}" data-owner="${item.image_id}">Open</button><button data-delete-object="${esc(item.id)}" data-owner="${item.image_id}" title="Delete">×</button></div></div></article>`).join("")||`<p class="muted">No objects in this class.</p>`;document.querySelectorAll("[data-open-object]").forEach(button=>button.onclick=()=>openEditor(button.dataset.owner,button.dataset.openObject));document.querySelectorAll("[data-object-class]").forEach(select=>select.onchange=()=>editObject(select.dataset.owner,select.dataset.objectClass,a=>a.class_id=+select.value));document.querySelectorAll("[data-delete-object]").forEach(button=>button.onclick=()=>{if(confirm("Delete this annotation?"))editObject(button.dataset.owner,button.dataset.deleteObject,null)})}
async function editObject(imageId,id,change){const detail=await api(`/api/v1/images/${imageId}`);if(change)change(detail.annotations.find(a=>a.id===id));else detail.annotations=detail.annotations.filter(a=>a.id!==id);await api(`/api/v1/images/${imageId}/annotations`,{method:"PUT",body:JSON.stringify({annotations:detail.annotations})});await Promise.all([loadImages(),loadObjects(),loadIssues()])}

async function loadIssues(){if(!state.meta)return;const data=await api("/api/v1/issues");$("issue-count").textContent=data.items.length?`(${data.items.length})`:"";$("issues").innerHTML=data.items.map(item=>`<div class="issue"><span class="pill">${esc(item.kind.replaceAll("_"," "))}</span><div><strong>${esc(item.image_name)}</strong><div class="muted">${esc(item.message)}</div></div><span class="muted">${esc(item.split||"")}</span><span>${item.image_id?`<button data-issue-open="${item.image_id}" data-object="${esc(item.annotation_id||'')}">Open</button>`:''}${item.fixable?` <button data-fix="${item.id}">Fix</button>`:''}</span></div>`).join("")||`<p class="muted">No issues found.</p>`;document.querySelectorAll("[data-issue-open]").forEach(button=>button.onclick=()=>openEditor(button.dataset.issueOpen,button.dataset.object||null));document.querySelectorAll("[data-fix]").forEach(button=>button.onclick=async()=>{try{await api(`/api/v1/issues/${button.dataset.fix}/fix`,{method:"POST"});await Promise.all([loadImages(),loadObjects(),loadIssues()])}catch(error){toast(error.message)}})}
function pointInPolygon(x,y,points){let inside=false;for(let i=0,j=points.length-2;i<points.length;j=i,i+=2)if((points[i+1]>y)!=(points[j+1]>y)&&x<(points[j]-points[i])*(y-points[i+1])/(points[j+1]-points[i+1])+points[i])inside=!inside;return inside}
function debounce(fn,wait){let timer;return(...args)=>{clearTimeout(timer);timer=setTimeout(()=>fn(...args),wait)}}
init().catch(error=>toast(error.message));
