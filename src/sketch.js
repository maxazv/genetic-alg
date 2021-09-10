var p;

function setup() {
  p = new Particle(width/2, height/2);
  vel = createVector(5, 2);
  p.vel = vel;
  grav = createVector(0, 0.01);
  friction = 0.97
  createCanvas(400, 400);
}

function draw() {
  background(45);
  p.show();
  p.move();
  if(p.edge()){
    print(p.vel);
    p.vel.mult(friction);
  }
  p.applyForce(grav)
}

class Particle{
  constructor(x, y){
    this.x = x;
    this.y = y;
    this.pos = createVector(this.x, this.y);
    this.vel = createVector(0, 0);
    this.acc = createVector(0, 0);

    this.r = 15;
  }
  move(){
    this.pos.x += this.vel.x;
    this.pos.y += this.vel.y;
  }
  applyForce(force){
    this.acc.add(force);
    this.vel.add(this.acc)
  }
  show(){
    noStroke();
    fill(255, 255, 255, 160);
    ellipse(this.pos.x, this.pos.y, this.r);
  }
  edge(){
    if(this.pos.x+this.r/2 > width){
      this.pos.x = width-this.r/2;
      this.vel.x *= -1;
      return true;
    }
    if(this.pos.y+this.r/2 > height){
      this.pos.y = height-this.r/2;
      this.vel.y *= -1;
      return true;
    }
    if(this.pos.x-this.r/2 < 0){
      this.pos.x = 0+this.r/2;
      this.vel.x *= -1;
      return true;
    }
    if(this.pos.y-this.r/2 < 0){
      this.pos.y = 0+this.r/2;
      this.vel.y *= -1;
      return true;
    }
    return false;
  }
}
