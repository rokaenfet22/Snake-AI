from os import stat_result
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from random import randint
import math
import pygame

"""
States: [x,y,food_x,food_y,up,down,left,right,obj_up,obj_down,obj_left,obj_right,food_up,food_down,food_left,food_right,food_upleft,food_upright,food_downleft,food_downright]
Actions: [0,1,2,3] up down left right
Rewards: 3 for catching food, -100 for dying, 1 for nearing food, -1 for further food

Environment: 8x8 grid
"""

a_to_s=[[0,-1],[0,1],[-1,0],[1,0]]
play_area_size=6
base_dirs=a_to_s

#visualizing using pygame
def visualizer(record): #record=[[s,snek_bod],...]
    win_dim=700
    px=win_dim//(play_area_size+2) 
    pygame.init()
    window=pygame.display.set_mode((win_dim,win_dim))
    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)
    a=record.pop(0)
    food=a[0][2:4]
    snek_bod=a[1]
    #print(snek_bod)
    auto=True
    run=True
    while run:
        if auto:
            pygame.time.delay(50)
        for event in pygame.event.get():
            if event.type==pygame.KEYUP:
                if event.key==pygame.K_SPACE:
                    auto=not auto
        if auto:
            if len(record)==0:
                run=False
            else:
                a=record.pop(0)
                food=a[0][2:4]
                snek_bod=a[1]
        window.fill(pygame.Color("black"))
        for i,n in enumerate(snek_bod): #snake
            red_grad=255-(i)*(255/len(snek_bod))
            pygame.draw.rect(window,((red_grad,0,0)),(n[0]*px,n[1]*px,px,px),0)
            draw_rect([n[0]*px,n[1]*px,px,px],"gray",1)
        if len(record)==0:
            draw_rect([snek_bod[0][0]*px,snek_bod[0][1]*px,px,px],"yellow",0)
        draw_rect([food[0]*px,food[1]*px,px,px],"green",0) #food
        pygame.display.flip()
    pygame.quit()

#network class
class QFunction(chainer.Chain):
    def __init__(self,obs_size,n_actions,n_hidden_channels=128):
        super(QFunction,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l4=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l5=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False):
        #activation function value
        h1=F.tanh(self.l1(x))
        h2=F.tanh(self.l2(h1))
        h3=F.tanh(self.l3(h2))
        h4=F.tanh(self.l4(h3))
        #classification
        y=chainerrl.action_value.DiscreteActionValue(self.l5(h4))
        return y

#random func
def random_action():
    return np.random.choice([0,1,2,3])

def get_dist(s):
    return math.sqrt((s[0]-s[2])**2+(s[1]-s[3])**2)

#find state and reward from cur state and action
def step(s,a,food_pos,snek_bod,dist):
    s=np.ndarray.tolist(s) #for easier handling
    r=0
    alive=True

    #move 4,5,6,7 (can't move in opposite of cur dir)
    if s[4] and a==1: a=0
    elif s[5] and a==0: a=1
    elif s[6] and a==3: a=2
    elif s[7] and a==2: a=3
    else: #update dir state
        ak=[0,0,0,0]
        ak[a]=1
        s[4:8]=ak
    s[0]+=a_to_s[a][0]
    s[1]+=a_to_s[a][1]

    #death check
    if s[0]>play_area_size or s[0]<0 or s[1]>play_area_size or s[1]<0 or (s[:2] in snek_bod):
        alive=False
        new_dist=0
    snek_bod.insert(0,s[:2])

    #check for food eating 2,3
    if s[:2]==food_pos:
        r=3
        while food_pos in snek_bod:
            food_pos=[randint(0,play_area_size),randint(0,play_area_size)]
        s[2:4]=food_pos #update food pos in state
        new_dist=get_dist(s)
    else:
        snek_bod.pop()
        #dist reward
        new_dist=get_dist(s)
        if new_dist<dist: r=1
        else: r=-1

    #obj_dir state update 8,9,10,11
    s[8:12]=[0,0,0,0]
    [x,y]=s[:2]
    #[0,-1]
    if y-1<0 or [x,y-1] in snek_bod: s[8]=1
    #[0,1]
    if y+1>play_area_size or [x,y+1] in snek_bod: s[9]=1
    #[-1,0]
    if x-1<0 or [x-1,y] in snek_bod: s[10]=1
    #[1,0]
    if x+1>play_area_size or [x+1,y] in snek_bod: s[11]=1

    #food_dir state updated 12,13,14,15
    s[12:16]=[0,0,0,0]
    if s[0]==s[2]:
        if s[1]>s[3]:
            s[12]=1
        else:
            s[13]=1
    if s[1]==s[3]:
        if s[0]>s[2]:
            s[14]=1
        else:
            s[15]=1
    #diag food_dir 16,17,18,19
    s[16:20]=[0,0,0,0]
    if s[2]-s[0]==s[3]-s[1]:
        if s[2]>s[0]:
            s[19]=1
        elif s[0]>s[2]:
            s[16]=1
    elif s[2]-s[0]==s[1]-s[3]:
        if s[2]>s[0]:
            s[17]=1
        elif s[0]>s[2]:
            s[18]=1

    #rough food dir 20,21,22,23
    if s[0]==s[2]:
        s[22]=s[23]=1
    elif s[0]>s[2]:
        s[22]=1
    else:
        s[23]=1
    if s[1]==s[3]:
        s[20]=s[21]=1
    elif s[1]>s[3]:
        s[20]=1
    else:
        s[21]=1

    #death reward upd
    if not alive: r=-100
    return np.array(s),r,alive,food_pos,snek_bod,new_dist

def train(num_episodes):
    #setting up agent
    gamma=0.95
    #num_episodes=300

    q_func=QFunction(16,4)
    optimizer=chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0,end_epsilon=0.01,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
    phi=lambda x: x.astype(np.float32,copy=False)
    agent=chainerrl.agents.DQN(q_func,optimizer,replay_buffer,gamma,explorer,replay_start_size=500,minibatch_size=500,update_interval=1,target_update_interval=100,phi=phi)
    #agent.load("snek_agent") #load existing agent

    #training
    longest_length=1
    for eps in range(num_episodes):
        record=[]
        #snek param
        snek_bod=[[0,0]]
        food_pos=[randint(1,play_area_size),randint(1,play_area_size)]
        #init
        s=np.array([0,0,food_pos[0],food_pos[1],0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1]) #facing down init
        if s[0]==s[2]:
            s[13]=1
        elif s[1]==s[3]:
            s[15]=1
        if s[2]==s[3]:
            s[19]=1
        R=0
        r=0
        alive=True
        dist=get_dist(s)
        while alive:
            record.append([s,snek_bod[:]])
            #get a and train
            a=agent.act_and_train(s[4:20],r)
            next_s,r,alive,food_pos,snek_bod,dist=step(s,a,food_pos,snek_bod,dist) #find next state from state and action
            #print(f"snek_pos={s[:2]}, food_pos={s[2:4]}, snek_dir={s[4:8]}, obj_dir={s[8:12]}, food_dir={s[12:20]} a={a}, r={r}, len={len(snek_bod)}")
            R+=r
            s=next_s #update state
        record.append([s,snek_bod[:]])
        agent.stop_episode_and_train(s[4:20],r,True) #stop and finalize training
        print(f"episode: {eps}, length: {len(snek_bod)}")
        if len(snek_bod)>longest_length:
            longest_length=len(snek_bod)
            visualizer(record)
        print(f"longest_len={longest_length}")
            
    print(f"longest_length: {longest_length}")
    #agent.save("snek_agent") #save agent

train(500) #train eps