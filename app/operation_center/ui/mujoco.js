const clamp = (value,min,max) => Math.max(min,Math.min(max,value));

export class MujocoView {
  constructor(image,onCamera,onHeading) {
    this.image=image;this.onCamera=onCamera;this.onHeading=onHeading;
    this.camera={azimuth:0,elevation:-15,distance:2.8};
    image.classList.add('simulation-controls','launcher-ready');image.dataset.renderer='mujoco';
    this.down=(event)=>{
      image.classList.add('pointer-control');
      if(![1,2].includes(event.button)) return;
      event.preventDefault();image.focus({preventScroll:true});image.setPointerCapture(event.pointerId);
      this.drag={id:event.pointerId,button:event.button,x:event.clientX,y:event.clientY};
      image.classList.add(event.button===1?'looking':'turning');
      if(event.button===2)this.onHeading(this.camera.azimuth*Math.PI/180);
    };
    this.move=(event)=>{
      if(!this.drag || event.pointerId!==this.drag.id) return;
      const dx=event.clientX-this.drag.x,dy=event.clientY-this.drag.y;
      this.camera.azimuth=((this.camera.azimuth-dx*.3+180)%360+360)%360-180;
      this.camera.elevation=clamp(this.camera.elevation-dy*.3,-85,5);
      this.onCamera({...this.camera});
      if(this.drag.button===2)this.onHeading(this.camera.azimuth*Math.PI/180);
      this.drag.x=event.clientX;this.drag.y=event.clientY;
    };
    this.cancel=()=>{
      if(this.drag && image.hasPointerCapture(this.drag.id))image.releasePointerCapture(this.drag.id);
      this.drag=null;image.classList.remove('looking','turning');
    };
    this.up=(event)=>{if(this.drag?.id===event.pointerId)this.cancel();};
    this.wheel=(event)=>{
      event.preventDefault();this.camera.distance=clamp(this.camera.distance*Math.exp(event.deltaY*.001),.75,8);
      this.onCamera({...this.camera});
    };
    this.aux=(event)=>event.preventDefault();
    this.keyboard=(event)=>{if(event.key==='Tab')image.classList.remove('pointer-control');};
    this.retry=()=>{if(this.visible)this.timer=setTimeout(()=>this.connect(),250);};
    image.addEventListener('pointerdown',this.down);
    image.addEventListener('pointermove',this.move);
    image.addEventListener('pointerup',this.up);
    image.addEventListener('pointercancel',this.up);
    image.addEventListener('wheel',this.wheel,{passive:false});
    image.addEventListener('auxclick',this.aux);
    image.addEventListener('error',this.retry);
    window.addEventListener('blur',this.cancel);
    window.addEventListener('keydown',this.keyboard);
  }

  get loaded() {return this.image.naturalWidth>0;}

  point(clientX,clientY) {
    if(!this.loaded)return null;
    const rect=this.image.getBoundingClientRect(),scale=Math.min(rect.width/this.image.naturalWidth,rect.height/this.image.naturalHeight);
    const width=this.image.naturalWidth*scale,height=this.image.naturalHeight*scale;
    const x=(clientX-rect.left-(rect.width-width)/2)/width,y=(clientY-rect.top-(rect.height-height)/2)/height;
    return x>=0 && x<=1 && y>=0 && y<=1 ? [x,y] : null;
  }

  connect() {if(this.visible)this.image.src=`/api/simulation/stream?t=${Date.now()}`;}

  setVisible(visible) {
    if(this.visible===visible)return;
    this.visible=visible;clearTimeout(this.timer);
    if(visible)this.connect();
    else {this.cancel();this.image.removeAttribute('src');}
  }

  destroy() {
    this.setVisible(false);
    for(const [type,handler] of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['auxclick',this.aux],['error',this.retry]])this.image.removeEventListener(type,handler);
    window.removeEventListener('blur',this.cancel);
    window.removeEventListener('keydown',this.keyboard);
  }
}
