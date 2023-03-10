#!/usr/bin/python3
import gym
import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from gazebo_connection import GazeboConnection
from gym import spaces, wrappers
import numpy as np 
from gym.utils import seeding
from tf.transformations import euler_from_quaternion

class QuadrotorEnv(gym.Env):

    def __init__(self):

        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.max_incl = np.radians(30)
        # nel caso migliore con velocità massima 1 m/s il drone dovrebbe impiegare circa 2 secondi per raggiungere ogni waypoints
        # per 6 waypoint sarebbero 12 secondi, considerando un dt di 0.1 secondi, il drone dovrebbe impiegare 120 step totali per terminare la traiettoria
        # poichè il controllo è subottimale rimango conservativo dando la possibiltà al drone di raggiungere il goal in 20 secondi che sarebbero 200 step
        self.time_step = 0.1
        self.current_step = 0
        self.step_limit = 200
        self.reward = 0.0

        # voglio che il drone decolli, compia una traiettoria quadrata e atterri nel punto di partenza
        self.waypoints = np.array([
                                [0, 0, 2],
                                [2, 0, 2],
                                [2, 2, 2],
                                [0, 2, 2],
                                [0, 0, 2],
                                [0, 0, 0]   
                                            ])
        self.waypoints_index = 0
        self.waypoint_threshold = 0.1
        self.traj_threshold = 0.1
        self.current_waypoint = self.waypoints[self.waypoints_index]
        self.next_waypoint = self.waypoints[(self.waypoints_index+1)%len(self.waypoints)]

        self.gazebo = GazeboConnection()

        # imposto uno stato delle azioni continuo, che siano le velocità lineari nella direzioni dei tre assi x y z e la velocita angolare intorno all'asse z
        self.action_space = spaces.Box(low = -1, high = 1, shape=(4,), dtype=np.float32)
        # voglio osservare: la distanza dal waypoint corrente, la distanza dal segmento che unisce due waypoints
        # in modo che il drone si mantegna vicino alla traiettoria, le velocità lineari, le velocità angolari,
        # gli angoli con i tre assi RPY
        self.observation_space = spaces.Box(low= -np.inf, high= np.inf, shape=(11,), dtype = np.float32) 
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        #self.init_drone()
        self.get_observation()
        self.waypoint_index = 0
        self.current_waypoint = self.waypoints[self.waypoints_index]
        self.next_waypoint = self.waypoints[(self.waypoints_index+1)%len(self.waypoints)]
        self.compute_distance_to_target()
        self.compute_distance_to_traj()
        observation = np.array([self.distance_to_waypoint, self.distance_to_line, 
                                self.vx, self.vy, self.vz,
                                self.p, self.q, self.r,
                                self.phi, self.theta, self.psi])
        self.reward = 0.0
        self.gazebo.pauseSim()

        return observation
    
    def step(self, action):
        self.gazebo.unpauseSim()
        # ad ogni azione eseguita aumenta il contatore degli step
        self.current_step += 1
        self.give_velocity_to_drone(action)
        # aspetto per uno step e vedo che succede
        rospy.sleep(self.time_step)
        # ricevo l'osservazione dall'environment
        self.get_observation()
        # metto in pausa la simulazione e calcolo la reward
        #self.gazebo.pauseSim()
        self.reward, done = self.get_reward()
        # le due distanze sono calcolate nell get_reward()
        observation = np.array([self.distance_to_waypoint, self.distance_to_line, 
                                self.vx, self.vy, self.vz,
                                self.p, self.q, self.r,
                                self.phi, self.theta, self.psi])
        return observation, self.reward, done, {}
    
    def bad_moves(self):
        wrong_roll = not(-self.max_incl < self.phi < self.max_incl)
        wrong_pitch = not(-self.max_incl < self.theta < self.max_incl)
        return wrong_roll, wrong_pitch

    def get_reward(self):
        wrong_roll, wrong_pitch = self.bad_moves()
        # calcolo la distanza tra la posizione del cdm del drone e il waypoint corrente
        self.compute_distance_to_target()
        # calcolo la distanza tra la posizione del cdm del drone e la traiettoria 
        self.compute_distance_to_traj() 
        rospy.loginfo('Distanza dal target:{} Distanza dalla traiettoria:{}'.format(self.distance_to_waypoint, self.distance_to_line))
        # la ricompensa sarà meno negativa quanto più si avvicina il drone al target e alla traiettoria
        self.reward = - self.distance_to_waypoint - self.distance_to_line
        # controllo se la distanza tra il drone e la traiettoria è minore della soglia 
        if self.distance_to_line < self.traj_threshold:
            self.reward += 1
            # controllo se la distanza tra il drone e il waypoint è minore della soglia, se è minore aggiorno l'indice del vettore waypoint
            if self.distance_to_waypoint < self.waypoint_threshold:
                self.reward += 10
                self.waypoints_index += 1
                rospy.loginfo('Moving to the next waypoint')
                # quando ho visitato tutti i waypoint resetto l'indice e faccio terminare l'episodio dando un bonus 
                if self.waypoints_index >= len(self.waypoints):
                    self.waypoints_index = 0
                    done = True
                    self.reward += 100
                    rospy.loginfo('Goal reached')
        # se il numero di step supera il limite termino l'episodio
        if self.current_step > self.step_limit:
            done = True
            #rospy.loginfo('Too much steps, you can do better than this!')
        # se il drone si inclina troppo, oppure se crasha, quindi si allontana troppo dal target o dalla traiettoria do una punizione e faccio terminare l'episodio
        elif wrong_roll or wrong_pitch or self.distance_to_waypoint > 2 or self.distance_to_line > 2:
            done = True
            self.reward -= 1000
            #rospy.loginfo('Your moves are very bad!')
        else:
            # per ogni step che esegue senza crashare do un bonus di 1 
            self.reward +=1
            done = False

        return self.reward, done
    
    def compute_distance_to_target(self):
        drone_position = np.array([self.x, self.y, self.z])
        self.distance_to_waypoint = np.linalg.norm(drone_position - self.waypoints[self.waypoints_index])

    def compute_distance_to_traj(self):
        drone_position = np.array([self.x, self.y, self.z])
        # voglio misurare la distanza tra il cdm del drone e il segmento che unisce due waypoints consecutivi 
        waypoints_length = np.linalg.norm(self.next_waypoint - self.current_waypoint)
        # se la distanza è nulla ritorno la distanza dal punto al waypoint corrente
        if waypoints_length == 0:
            return np.linalg.norm(drone_position-self.current_waypoint)
        drone_vector = drone_position - self.current_waypoint
        waypoints_vector = self.next_waypoint - self.current_waypoint
        # unit vector in direzione di weypoint_vectori,per calcolare la proiezione di drone_vector sulla traiettoria 
        w_vec_unit = waypoints_vector/waypoints_length
        # se la proiezione del cdm del drone si trova all'esterno del segmento che unisce i due waypoints 
        # ritorno la distanza dal waypoints più vicino
        t = np.dot(waypoints_vector, w_vec_unit) / (waypoints_length**2)
        if t < 0:
            closest_point = self.current_waypoint
        elif t > 1:
            closest_point = self.next_waypoint
        else:
            # proeizione è un vettore dal primo endpoint della traiettoria al punto più vicino sulla traittoria al cdm del drone
            projection = np.dot(drone_vector, w_vec_unit) * w_vec_unit
            closest_point = self.current_waypoint + projection
        # la distanza tra il cdm del drone e il punto più vicino sulla traiettoria
        self.distance_to_line = np.linalg.norm(drone_position - closest_point)
         
    def get_observation(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        resp = get_state(model_name = "quadrotor")
        self.z = resp.pose.position.z
        self.x = resp.pose.position.x
        self.y = resp.pose.position.y
        orientation = [resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w]
        self.phi, self.theta, self.psi = euler_from_quaternion(orientation) # roll, pitch, yaw
        self.vz = resp.twist.linear.z
        self.vx = resp.twist.linear.x
        self.vy = resp.twist.linear.y
        self.p = resp.twist.angular.x
        self.q = resp.twist.angular.y
        self.r = resp.twist.angular.z
    
    def init_drone(self):
        state_msg = ModelState()
        state_msg.model_name = "quadrotor"
        state_msg.pose.position.x = 0
        state_msg.pose.position.y = 0
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        set_state(state_msg)
    
    def give_velocity_to_drone(self, action):
        vel_msg = Twist()
        vel_msg.linear.x = action[0] # [-1, 1] m/s
        vel_msg.linear.y = action[1] # [-1, 1] m/s
        vel_msg.linear.z = action[2]
        vel_msg.angular.z = action[3] # [-1, 1] m/s
        self.vel_pub.publish(vel_msg)
        
    def map_action(self,x, in_min, in_max, out_min, out_max):
	    return ((((x - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min)
