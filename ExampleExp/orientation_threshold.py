#!/usr/bin/env python
# -*- coding: utf-8 -*-

# You must have a monitor in the monitor center of Psychopy!!

# import modules

from __future__ import absolute_import, division
import os  # handy system and path functions
import sys  # to get file system encoding
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
import math
import numpy as np
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import pandas as pd
from sklearn.cluster import KMeans
from webcolors import name_to_rgb
from psychopy.hardware import keyboard
from psychopy.misc import fromFile


###########################################################################

# Some functions

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def good_fix(fix, gaze, tolerance):
    # all in the same unit
    if distance(fix, gaze) <= tolerance:
        return True
    else:
        return False

def create_mndnp(win, color, cursor_size = (0.1, 0.1)):
    mouse = event.Mouse(visible = False)
    dot = visual.GratingStim(win = win, tex= None, units = 'deg', 
                             mask="circle", pos=(0, 0), size=cursor_size, colorSpace = 'rgb255', 
                             color=color, autoLog=False)
    dot_pos = visual.TextStim(win = win, pos=(-1,-1), 
                              units = 'norm', alignHoriz='left', alignVert = 'bottom', height=0.06, 
                              text='', autoLog=False)
    return mouse, dot, dot_pos

def create_fixation(win, pos, size, color):
    fixation = visual.RadialStim(win = win, units='deg', pos=pos,
                                size=size, radialCycles=0.6, angularCycles=0, radialPhase=-0.4, angularPhase=0,
                                ori=0, texRes=64, angularRes=100, 
                                color=color, colorSpace='rgb255',
                                contrast=1.0, opacity=1.0)
    return fixation

def create_text(win, color, text = ''):
    textstim = visual.TextStim(win=win, text=text, font='Arial',
                               units='norm', pos=(0,0), height=0.08,
                               wrapWidth=None, ori=0, color=color, 
                               colorSpace='rgb255', opacity=1, languageStyle='LTR')
    return textstim

def create_info(win):
    information = visual.TextStim(win = win, pos=(-1, 1), 
                           units = 'norm', alignHoriz='left', alignVert = 'top', 
                           height=0.04, text='', autoLog=False)
    return information

def non_rep_append(l, ele):
    if l:
        if l[-1] == ele:
            pass
        else:
            l.append(ele)
    else:
        l.append(ele)
    return l

def click_recorder(mouse, mouse_x, mouse_y, dots, win, eye_tracker = False):
    beep = sound.Sound('300', volume = 0.1, secs=0.3, stereo=True, hamming=False)
    mouse0, mouse1, mouse2 = mouse.getPressed()
    record = True
    if eye_tracker:
        record = good # good is a global var, indicates whether or not the participant is gazing at the fixation object.
    if mouse0 and record:
        non_rep_append(dots, (mouse_x, mouse_y))
        beep.play(when = win)
        mouse.clickReset()
        return True
    else:
        return False

def dots_estimation(dots, axis):
    # axis should be set 0 for HT and HD data, 1 for VT data.
    def washing(data, axis):
        # group dots in the subset to 2 sub_groups
        kmeans = KMeans(2).fit(data)
        labels = kmeans.labels_
        sub1 = [data[x] for x in range(len(labels)) if labels[x] == 0]
        sub2 = [data[x] for x in range(len(labels)) if labels[x] == 1]
        # exclude outliers (1.5 IQR)
        sub1_q3, sub1_q1 = np.percentile(sub1, [75, 25], axis = 0)
        sub1_iqr = (sub1_q3 - sub1_q1)[axis]
        sub1_interval = [sub1_q1[axis] - 1.5 * sub1_iqr, sub1_q3[axis] + 1.5 * sub1_iqr]
        sub2_q3, sub2_q1 = np.percentile(sub2, [75, 25], axis = 0)
        sub2_iqr = (sub2_q3 - sub2_q1)[axis]
        sub2_interval = [sub2_q1[axis] - 1.5 * sub2_iqr, sub2_q3[axis] + 1.5 * sub2_iqr]
        new_sub1 = [dot for dot in sub1 if abs(dot[axis] - sub1_interval[0]) + abs(dot[axis] - sub1_interval[1]) <= abs(sub1_interval[0] - sub1_interval[1]) ]
        new_sub2 = [dot for dot in sub2 if abs(dot[axis] - sub2_interval[0]) + abs(dot[axis] - sub2_interval[1]) <= abs(sub2_interval[0] - sub2_interval[1]) ]
        return new_sub1 + new_sub2

    def km_centers(km_model, axis):
        dot1, dot2 = km_model.cluster_centers_
        if dot1[axis] <= dot2[axis]:
            pass
        else:
            dot1, dot2 = dot2, dot1
        return tuple(dot1), tuple(dot2)
    
    # Calculate 2 center dots for dots.
    c_dots = []
    if len(dots) == 1:
        c_dots.append(dots[0])
    else:
        # data washing, filter out outliers
        dots = washing(dots, axis)
        kmeans = KMeans(2).fit(dots)
        dot1, dot2 = km_centers(kmeans, axis)
        c_dots.append(dot1)
        c_dots.append(dot2)
    return c_dots

def order_dots(dots):
    # compute the highest dot of these dots
    dots = np.array(dots)
    top = [dot for dot in dots if dot[1] == dots.max(axis = 0)[1]][0]
    # calculate degree to top dot for each dot in dots
    def deg(dot):
        arctan = math.atan2(*(dot - top)[::-1])
        return math.degrees(arctan) % 360
    # sort dots by degree
    ordered_dots = sorted(dots, key=deg)
    return [tuple(dot) for dot in ordered_dots]

def sum_bad_fixation(tracking_dict, onset_frameN, offset_frameN, time_back = 0.03, time_forward = 0, framedur = 1/60):
    try:
    # framedur is a global var
        frames_back = round(time_back / framedur) # convert time_back(in second) to frame_back(in frame count)
        frames_forward = round(time_forward / framedur) # convert time_forward(in second) to frame_forward(in frame count)
        start_idx = int(tracking_dict['frameN'].index(onset_frameN) - frames_back)
        end_idx = int(tracking_dict['frameN'].index(offset_frameN) + frames_forward)
        sumation = end_idx - start_idx - sum(tracking_dict['good_fixation'][start_idx:end_idx])
    except Exception as e:
        print(e)
        sumation = 1
    return sumation

###########################################################################


###########################################################################

# Routine functions

def border_points(win, exp_config, path, dimension):
    # dimension should be set to 0 for HT and HD, 1 for VT
    clicks = []

    StartClock = core.Clock()
    start_beep = sound.Sound('1000', secs=1.0, stereo=True, hamming=False, volume = 1)
    start_fixation = create_fixation(win = win, pos = exp_config['Fix_Pos(deg)'], 
                                     size = exp_config['Fix_Size(deg)'], color = exp_config['Fix_Color'])

    TrialsClock = core.Clock()
    trials_fixation = create_fixation(win = win, pos = exp_config['Fix_Pos(deg)'], 
                                      size = exp_config['Fix_Size(deg)'], color = exp_config['Fix_Color'])
    trials_mouse, trials_dot, trials_dot_pos = create_mndnp(win = win, color = exp_config['Cursor_Color'], 
                                                            cursor_size = exp_config['Cursor_Size(deg)'])
    trials_end = keyboard.Keyboard()
    trials_info = create_info(win = win)

    routineTimer = core.CountdownTimer()
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    start_autodraw = [start_fixation]
    # reset timers
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    StartClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    continueRoutine = True
    
    beep_start = False
    fixation_start = False

    # -------Starting Routine "Start"-------
    while continueRoutine and routineTimer.getTime() > 0:
        tThisFlip = win.getFutureFlipTime(clock=StartClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        # update/draw components on each frame
        # start/stop start_beep
        if not beep_start and tThisFlip >= 0.0-exp_config['FrameTolerance']:
            start_beep.tStartRefresh = tThisFlipGlobal  # on global time
            start_beep.play(when=win)  # sync with win flip
            beep_start = True
        if beep_start:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_beep.tStartRefresh + 1.0-exp_config['FrameTolerance']:
                start_beep.stop()
    
        # *start_fixation* updates
        if not fixation_start and tThisFlip >= 0.0-exp_config['FrameTolerance']:
            start_fixation.tStartRefresh = tThisFlipGlobal  # on global time
            start_fixation.setAutoDraw(True)
            fixation_start = True
        if fixation_start:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_fixation.tStartRefresh + 1.0-exp_config['FrameTolerance']:
                start_fixation.setAutoDraw(False)
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        elif continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    # -------Ending Routine "Start"-------
    for thisComponent in start_autodraw:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    start_beep.stop()  # ensure sound has stopped at end of routine

    # ------Prepare to start Routine "Trials"-------
    # keep track of which components have finished
    trials_autodraw = [trials_fixation, trials_dot, trials_dot_pos, trials_info]
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    TrialsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    if exp_config['Eye_Tracker']:
        trials_tracking = {'frameN': [], 'routine_time':[], 'gaze':[], 'good_fixation':[], 'click': []}
    else:
        trials_tracking = {'None': 'eye tracker not found'}

    # -------Run Routine "Trials"-------
    trials_mouse.clickReset()
    trials_dot_ocolor_rgb = [float(x) for x in trials_dot.color]
    trials_dot_rcolor_rgb = [float(255 - x) for x in trials_dot.color]
    keyboard_start = False
    while continueRoutine:
        # get current time
        t = TrialsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=TrialsClock)
        frameN += 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *trials_fixation*, *trials_mouse* and *trials_dot* updates
        if tThisFlip >= 0.0-exp_config['FrameTolerance']:
            if exp_config['Eye_Tracker']:
                gaze = tracker.getPosition()
                if gaze is None:
                    gaze = (99999, 99999)
                else: pass
                good = good_fix(exp_config['Fix_Pos'], gaze, exp_config['EyeTolerance'])
                if not good:
                    trials_fixation.color = tuple(name_to_rgb('red'))
                else:
                    trials_fixation.color = exp_config['Fix_Color']
                trials_tracking['frameN'].append(frameN)
                trials_tracking['routine_time'].append(t)
                trials_tracking['gaze'].append(gaze)
                trials_tracking['good_fixation'].append(good)
            trials_fixation.setAutoDraw(True)

            # Note that the flickering cursor switches its color according to frameN!!
            # so be cautious about actual framerate and the flickering rate you want
            if exp_config['Flickering(Hz)'] and frameN % exp_config['Flickering_FrameRate'] == 0:
                if all(np.array(trials_dot.color) == np.array(trials_dot_ocolor_rgb)):
                    trials_dot.color = trials_dot_rcolor_rgb
                elif all(np.array(trials_dot.color) == np.array(trials_dot_rcolor_rgb)):
                    trials_dot.color = trials_dot_ocolor_rgb
                else: pass
            else: pass
            xy = (path, trials_mouse.getPos()[dimension])
            mouse_x, mouse_y = xy[1-dimension], xy[dimension]
            trials_dot.setPos(newPos = [mouse_x, mouse_y])
            clickreport = click_recorder(trials_mouse, mouse_x, mouse_y, clicks, win = win, eye_tracker = exp_config['Eye_Tracker'])
            if exp_config['Eye_Tracker']:
                trials_tracking['click'].append(clickreport)
            trials_dot.setAutoDraw(True)

        # *trials_info*, and *trials_dot_pos* updates
        if tThisFlip >= 0.0-exp_config['FrameTolerance'] and exp_config['Show_Raw_Data']:
            trials_info.text = (f"Time: \n{t}\n\nClicks:\n{clicks}")
            trials_info.setAutoDraw(True)
            trials_dot_pos.text = f"X: {mouse_x}\nY: {mouse_y}"
            trials_dot_pos.setAutoDraw(True)

        # *trials_end* updates
        if not keyboard_start and tThisFlip >= 0.0-exp_config['FrameTolerance']:
            win.callOnFlip(trials_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
            keyboard_start = True
        if keyboard_start:
            theseKeys = trials_end.getKeys(keyList=['space', 'e', 'd'], waitRelease=False)
            if len(theseKeys):
                theseKeys = theseKeys[0]  # at least one key was pressed
                if theseKeys == 'd':
                    exp_config['Show_Raw_Data'] = not exp_config['Show_Raw_Data']
                    trials_info.setAutoDraw(exp_config['Show_Raw_Data'])
                    trials_dot_pos.setAutoDraw(exp_config['Show_Raw_Data'])
                elif theseKeys == 'space':
                    continueRoutine = False
                    dots = dots_estimation(clicks, axis = dimension)
                elif theseKeys == 'e' and exp_config['Eye_Tracker']:
                    win.winHandle.minimize()
                    win.winHandle.set_fullscreen(False)
                    tracker.setRecordingState(False)
                    tracker.runSetupProcedure()
                    tracker.setRecordingState(True)
                    win.winHandle.maximize()
                    win.winHandle.set_fullscreen(True)
                    win.winHandle.activate()
            else: pass
    
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
    
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break    
        # refresh the screen
        elif continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    if exp_config['Eye_Tracker']:
        trials_tracking = pd.DataFrame(trials_tracking)
        trials_tracking['Participant'] = exp_config['Participant']
        trials_tracking['Session'] = exp_config['Session']
    # -------Ending Routine "Trials"-------
    for thisComponent in trials_autodraw:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    return dots, trials_tracking


def orien_staircase(win, exp_config, ori_bs_pos):
    # Initialize components for Routine "Staircase"
    StaircaseClock = core.Clock()
    staircase_fixation = create_fixation(win = win, pos = exp_config['Fix_Pos(deg)'], size = exp_config['Fix_Size(deg)'], 
                                         color = exp_config['Fix_Color'])
    staircase_start_beep = sound.Sound('1000', secs=1.0, stereo=True, hamming=False, volume = 1)
    staircase_info = create_info(win = win)
    staircase_end = keyboard.Keyboard()
    staircase_inset = visual.GratingStim(win=win, sf=1, size=2.4, mask='circle', ori=0, units = 'deg') # vertical grating
    staircase_grating = visual.GratingStim(win=win, sf=1,  size=10, mask='circle', ori=exp_config['Reference_Orientation'],
                                           units = 'deg') # vertical grating
    staircase_doUCit = sound.Sound(value = 'doUCit.wav', secs = 1.05, stereo = True, hamming=False, volume = 1)
    side_code = np.random.randint(0, 2)
    if side_code == 0:
        choices = ['continuous', 'discontinuous']
    elif side_code == 1:
        choices = ['discontinuous', 'continuous']
    staircase_rating = visual.RatingScale(win = win, noMouse = True, pos=(0, 0), showValue = True, marker='circle', 
                                          size=0.85, name='up', choices = choices, markerStart= 0.5,
                                          scale = 'Discontinuous or Continuous?')

    # ------Prepare to start Routine "Staircase"-------
    # Theoretically, 
    # 2D1U, 3D1U, 4D1U, 5D1U, 6D1U staircases should target 
    # 0.71, 0.79, 0.84, 0.87, 0.89 thresholds, while 
    # the amount of trials influences std of threshold estimate.
    # The default setting is 3D1U staircase, but you can adjust it yourself.

    # update component parameters for each repeat
    staircase_autodraw = [staircase_fixation, staircase_info, staircase_inset, staircase_grating]

    # Specify potential fix positions.
    fix_pos_x = [int(x) for x in np.linspace(-6, 2, 10)] # x coordinates, here, the fixation is allowed to lay between -5 and 5 deg horizontally
    fix_pos_y = [int(x) for x in np.linspace(-2, 2, 10)] # y coordinate, here the fixation is allowed to lay between -2 and 2 deg vertically
    # This setting of fix pos fits 40cm by 30cm screen and distance = 57cm.
    fix_idx_range = len(fix_pos_x) # there should be 10 idx for x and y values, so there are in total 100 potential positions for the fixation.

    # Specify stimuli positions relative to the fix position.
    eccentricity = math.sqrt((ori_bs_pos[0])**2 + (ori_bs_pos[1])**2)
    rad_bs_norm = 2 * math.asin((0.5*np.array(staircase_grating.size).max(axis = 0))/eccentricity)
    rad_bs_hori = math.asin(ori_bs_pos[1] / eccentricity)
    rad_upper_hori = rad_bs_hori + rad_bs_norm
    rad_lower_hori = rad_bs_hori - rad_bs_norm
    ori_upper_pos = (eccentricity * math.cos(rad_upper_hori), eccentricity * math.sin(rad_upper_hori))
    ori_lower_pos = (eccentricity * math.cos(rad_lower_hori), eccentricity * math.sin(rad_lower_hori))

    event.clearEvents()
    frameN = -1
    if exp_config['Eye_Tracker']:
        staircase_tracking = {'frameN': [], 'routine_time':[], 'gaze':[], 'good_fixation':[], 'button':[], 'stimulus_display':[], 'recorded_response':[]}
    else:
        staircase_tracking = {'None': 'eye tracker not found'}

    # -------Run Routine "Staircase"-------
    ready = False
    offset = False
    # Note that the duration is according to frameN!!
    # so be cautious about the actual framerate and duration you want
    duration = exp_config['Staircase_Target_Display(s)']
    duration_frames = round(duration / exp_config['FrameDur'])
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    StaircaseClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    # a beep indicating the start of staircase procedure.
    beep_start = False
    if not beep_start:
        start_time = StaircaseClock.getTime()
        staircase_start_beep.play(when=win)  # sync with win flip
        beep_start = True
    while beep_start:
        # is it time to stop? (based on global clock, using actual start)
        if StaircaseClock.getTime() - start_time > 1:
            staircase_start_beep.stop()
            beep_start = False
        else: pass

    # staircase should not be added into StaircaseComponents
    staircase = data.StairHandler(startVal = exp_config['Start_Orientation_Contrast'], 
                              nReversals=exp_config['nReversals'], 
                              stepSizes=exp_config['Step_Sizes(ori)'], 
                              nTrials=exp_config['Staircase_Trials'], 
                              nUp=exp_config['nUp'], nDown=exp_config['nDown'], 
                              extraInfo=None, stepType='lin', minVal=0, maxVal=90)
    staircase_trials = 0

    for thisIncrement in staircase:
        staircase_trials += 1
        thisResp = None
        # reset timers
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        doUCit_start = False
        keyboard_start = False
        continueRoutine = True
        staircase_inset.setOri(staircase_grating.ori - thisIncrement)
        fix_pos = (fix_pos_x[np.random.randint(0, fix_idx_range)], fix_pos_y[np.random.randint(0, fix_idx_range)])
        staircase_fixation.setPos(fix_pos)
        bs_pos = (ori_bs_pos[0] + fix_pos[0], ori_bs_pos[1] + fix_pos[1])
        upper_pos = (ori_upper_pos[0] + fix_pos[0], ori_upper_pos[1] + fix_pos[1])
        lower_pos = (ori_lower_pos[0] + fix_pos[0], ori_lower_pos[1] + fix_pos[1])
        positions = [bs_pos, upper_pos, lower_pos]
        pos_idx = np.random.randint(0, 3)
        pos_labels = ['BS', 'Upper', 'Lower']
        pos_label = pos_labels[pos_idx]
        staircase_inset.setPos(positions[pos_idx])
        staircase_grating.setPos(positions[pos_idx])  # in other location
        while continueRoutine:
            t = StaircaseClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=StaircaseClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN += 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop staircase_doUCit
            if not doUCit_start and tThisFlip >= 0.0-exp_config['FrameTolerance'] and ready and offset:
                staircase_doUCit.tStartRefresh = tThisFlipGlobal  # on global time
                staircase_doUCit.play(when=win)  # sync with win flip
                doUCit_start = True
            if doUCit_start:
                if tThisFlipGlobal > staircase_doUCit.tStartRefresh + 1.05-exp_config['FrameTolerance']:
                    staircase_doUCit.stop()
        
            # *staircase_fixation*, *staircase_inset*, and *staircase_grating* updates
            if tThisFlip >= 0.0-exp_config['FrameTolerance']:
                if offset:
                    staircase_fixation.opacity = 0
                else:
                    staircase_fixation.opacity = 1
                if exp_config['Eye_Tracker']:
                    gaze = tracker.getPosition()
                    if gaze is None:
                        gaze = (99999, 99999)
                    else: pass
                    good = good_fix(exp_config['Fix_Pos(deg)'], gaze, exp_config['EyeTolerance'])
                    if not good:
                        staircase_fixation.color = tuple(name_to_rgb('red'))
                    else:
                        staircase_fixation.color = exp_config['Fix_Color']
                    staircase_tracking['frameN'].append(frameN)
                    staircase_tracking['routine_time'].append(t)
                    staircase_tracking['gaze'].append(gaze)
                    staircase_tracking['good_fixation'].append(good)
                staircase_fixation.setAutoDraw(True)

                if ready:
                    # if the participant triggers the target object, set it visible.
                    staircase_inset.opacity = 1
                    staircase_grating.opacity = 1
                    # Note that the flickering cursor switches its color according to frameN!!
                    # so be cautious about actual framerate and the flickering rate you want
                    if frameN <= offset_frameN:
                        pass
                    elif frameN > offset_frameN:
                        staircase_inset.opacity = 0
                        staircase_grating.opacity = 0
                        offset = True
                else:
                    staircase_inset.opacity = 0
                    staircase_grating.opacity = 0
                    offset = False
                staircase_grating.setAutoDraw(True)
                staircase_inset.setAutoDraw(True)

            # *staircase_info* updates
            if tThisFlip >= 0.0-exp_config['FrameTolerance'] and exp_config['Show_Raw_Data']:
                staircase_info.text = (f"Display duration(frames): {duration_frames}\nFrameN: {frameN}\n\n"
                                       f"Staircase_trials: {staircase_trials}\n"
                                       f"Ori_Contrast: {thisIncrement}\nPosition: {pos_label}\n\nTime: {t}")
                staircase_info.setAutoDraw(True)
            
            # *staircase_rating* updates
            if ready and offset:
                win.mouseVisible = True
                staircase_rating.noMouse = False
                staircase_rating.mouseOnly = True
                staircase_rating.setAutoDraw(True)
                if not staircase_rating.noResponse:
                    if pos_idx != 0 and not exp_config['Eye_Tracker'] or not sum_bad_fixation(staircase_tracking, onset_frameN, offset_frameN, time_back = 0.03, framedur = exp_config['FrameDur']):
                        rating = staircase_rating.getRating()
                        if rating == 'continuous':
                            thisResp = 0
                        elif rating == 'discontinuous':
                            thisResp = 1
                        continueRoutine = False
                    else:
                        continueRoutine = True
                        doUCit_start = False
                        fix_pos = (fix_pos_x[np.random.randint(0, fix_idx_range)], fix_pos_y[np.random.randint(0, fix_idx_range)])
                        staircase_fixation.setPos(fix_pos)
                        bs_pos = (ori_bs_pos[0] + fix_pos[0], ori_bs_pos[1] + fix_pos[1])
                        upper_pos = (ori_upper_pos[0] + fix_pos[0], ori_upper_pos[1] + fix_pos[1])
                        lower_pos = (ori_lower_pos[0] + fix_pos[0], ori_lower_pos[1] + fix_pos[1])
                        positions = [bs_pos, upper_pos, lower_pos]
                        pos_idx = np.random.randint(0, 3)
                        pos_labels = ['BS', 'Upper', 'Lower']
                        pos_label = pos_labels[pos_idx]
                        staircase_inset.setPos(positions[pos_idx])
                        staircase_grating.setPos(positions[pos_idx])  # in other location
                    win.mouseVisible = False
                    ready = False
                    offset = False
                    staircase_rating.setAutoDraw(False)
                    side_code = np.random.randint(0, 2)
                    if side_code == 0:
                        choices = ['continuous', 'discontinuous']
                    elif side_code == 1:
                        choices = ['discontinuous', 'continuous']
                    staircase_rating = visual.RatingScale(win = win, noMouse = True, pos=(0, 0), showValue = True, 
                                                          marker='circle', size=0.85, name='up', choices = choices, 
                                                          scale = 'Discontinuous or Continuous', markerStart=0.5)
                    core.wait(exp_config['Staircase_Latency(s)'])

            # *staircase_end* updates
            if not keyboard_start and tThisFlip >= 0.0-exp_config['FrameTolerance']:
                win.callOnFlip(staircase_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
                event.clearEvents()
                keyboard_start = True
            if keyboard_start:
                theseKeys = staircase_end.getKeys(keyList=['space', 'e', 'd'], waitRelease=False)
                if len(theseKeys):
                    theseKeys = theseKeys[0]  # at least one key was pressed
                    # check for quit:
                    if theseKeys == 'e' and exp_config['Eye_Tracker']:
                        theseKeys = theseKeys.name
                        win.winHandle.minimize()
                        win.winHandle.set_fullscreen(False)
                        tracker.setRecordingState(False)
                        tracker.runSetupProcedure()
                        tracker.setRecordingState(True)
                        win.winHandle.maximize()
                        win.winHandle.set_fullscreen(True)
                        win.winHandle.activate()
                    elif theseKeys == 'd':
                        theseKeys = theseKeys.name
                        exp_config['Show_Raw_Data'] = not exp_config['Show_Raw_Data']
                        staircase_info.setAutoDraw(exp_config['Show_Raw_Data'])
                    elif not ready:
                        if theseKeys == 'space':
                            theseKeys = theseKeys.name
                            if not exp_config['Eye_Tracker']:
                                continueRoutine = True
                                ready = True
                                onset_time = core.getTime()
                                onset_frameN = frameN # but the target will be displayed on the next frame.
                                offset_frameN = onset_frameN + duration_frames
                            elif good:
                                continueRoutine = True
                                ready = True
                                onset_time = core.getTime()
                                onset_frameN = frameN
                                offset_frameN = onset_frameN + duration_frames
                else:
                    theseKeys = 'None'
                if exp_config['Eye_Tracker']:
                    staircase_tracking['button'].append(theseKeys)
                    staircase_tracking['stimulus_display'].append(ready and not offset)
                    staircase_tracking['recorded_response'].append(str(thisResp))
                    

            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
    
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break  
            # refresh the screen
            elif continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        staircase.addResponse(thisResp)
        continueRoutine = True
        if staircase_trials < staircase.nTrials:
            pass
        elif staircase_trials == staircase.nTrials:
            mean_final_n_trials = np.average(staircase.intensities[-exp_config['Final_n_Trials']:])
    try:
        staircase.saveAsPickle(exp_config['FileName'] + 'Staircase') # this file can be analyzed by psychopy official demo
    except Exception: pass

    if exp_config['Eye_Tracker']:
        staircase_tracking = pd.DataFrame(staircase_tracking)
        staircase_tracking['Participant'] = exp_config['Participant']
        staircase_tracking['Session'] = exp_config['Session']

    # -------Ending Routine "Staircase"-------
    for thisComponent in staircase_autodraw:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    win.flip()
    return mean_final_n_trials, staircase_tracking


###########################################################################


###########################################################################

# -------Prepare experiment-------

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'exp'

# Exp settings, can be specified before experiment
expInfo = {'Session': 'test', 'Participant': 'test', 'Sex': 'male', 'Age': '24', 
           'Monitor_Name': 'testMonitor', 'Monitor_Size(pix)': '1920, 1080', 'Monitor_Size(cm)': '31, 17.5', 
           'Distance(cm)': '28',
           'BG_Color': '128, 128, 128', 'Fix_Color': 'white', 'Fix_Size(deg)': '1.26', 'Fix_Pos(deg)': '9.34, 0', 
           'Cursor_Color': 'white', 'Cursor_Size(deg)': '0.38', 'Flickering(Hz)': '20', 
           'Test_Eye': ['RIGHT', 'LEFT'], 
           'Horizontal_Trials': 'Default', 'Vertical_Trials': 'Default',
           'Staircase': ['True', 'False'],
           'Eye_Tracker': ['True', 'False']}
# Apart from 'Default', 'Horizontal_Trials' and 'Vertical_Trials' accept a list or tuple containing ints or floats.
# e.g., [1]
expdlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if expdlg.OK == False: # "dlg.OK" returns boolean. "dlg.OK" == bool(dlg)
    core.quit()  # user pressed cancel

# Staircase settings
if expInfo['Staircase'] == 'True':
    staircaseInfo = {'Reference_Orientation': '90', 'Start_Orientation_Contrast': '50', 
                     'Step_Sizes(ori)': '10, 5, 5, 2, 2, 1, 1', 
                     'Staircase_Trials': '50', 'nUp': '1', 'nDown': '2', 'nReversals': '8', 
                     'Final_n_Trials': '10',
                     'Staircase_Target_Display(s)': '0.4', 'Staircase_Latency(s)': '0.4'}
    staircasedlg = gui.DlgFromDict(dictionary = staircaseInfo, sortKeys = False, title = 'Staircase settings')
    if staircasedlg.OK == False:
        core.quit()
elif expInfo['Staircase'] == 'False':
    staircaseInfo = {'Reference_Orientation': '90', 'Threshold_Orientation_Contrast': 'None'}
    staircasedlg = gui.DlgFromDict(dictionary = staircaseInfo, sortKeys = False, title = 'Staircase settings')
    if staircasedlg.OK == False:
        core.quit()

# Eye tracker settings
if expInfo['Eye_Tracker'] == 'True':
    eyetrackingInfo = {'Tolerance(deg)': '1.5', 'Calibration_Type': ['HV9']} # 9-point calibration is good enough
    eyetrackingdlg = gui.DlgFromDict(dictionary = eyetrackingInfo, sortKeys = False, title = 'Eyetracker settings')
    if eyetrackingdlg.OK == False:
        core.quit()
    for key in eyetrackingInfo:
        try:
            eyetrackingInfo[key] = eval(eyetrackingInfo[key])
        except: pass
elif expInfo['Eye_Tracker'] == 'False':
    pass

# Use eval() to transform str to other types like list and tuple.
for key in staircaseInfo:
    staircaseInfo[key] = eval(staircaseInfo[key])


# Some exp info is to be used, so transform str to correct types.
for key in ['Monitor_Size(pix)', 'Monitor_Size(cm)', 'Distance(cm)',
           'BG_Color', 'Fix_Color', 'Fix_Size(deg)', 'Fix_Pos(deg)', 
           'Cursor_Color', 'Cursor_Size(deg)', 'Flickering(Hz)',
           'Staircase', 'Eye_Tracker']:
    try:
        expInfo[key] = eval(expInfo[key])
    except Exception:
        expInfo[key] = tuple(name_to_rgb(expInfo[key]))

expInfo['Fix_Pos(deg)'] = list(expInfo['Fix_Pos(deg)'])

if expInfo['Test_Eye'] == 'RIGHT':
    a = -1
elif expInfo['Test_Eye'] == 'LEFT':
    a = 1
else:
    a = 1
expInfo['Fix_Pos(deg)'][0] = a * abs(expInfo['Fix_Pos(deg)'][0])
expInfo['Date'] = data.getDateStr()  # add a simple timestamp
expInfo['ExpName'] = expName

# Start Code - component code to be run before the window creation
# Eye tracker (if set True)
# This eye tracker uses the same units (deg here) with win
if expInfo['Eye_Tracker']:
    from psychopy.iohub import launchHubServer
    iohub_config = {'eyetracker.hw.sr_research.eyelink.EyeTracker':
                {'name': 'tracker',
                 'model_name': 'EYELINK 1000 DESKTOP',
                 'runtime_settings': {'sampling_rate': 1500,
                                      'track_eyes': expInfo['Test_Eye']}
                 }
                }
    io = launchHubServer(**iohub_config)
    # Get the eye tracker device.
    tracker = io.devices.tracker
    tracker.runSetupProcedure()
    tracker.setRecordingState(True)

# Setup the Window
win = visual.Window(
    size=expInfo['Monitor_Size(pix)'], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor=expInfo['Monitor_Name'], colorSpace='rgb255', color=expInfo['BG_Color'], 
    blendMode='avg', useFBO=True, units='deg')
frameTolerance = 0.001  # how close to onset before 'same' frame
# Store frame rate of monitor if we can measure it
expInfo['FrameRate'] = round(win.getActualFrameRate())
if expInfo['FrameRate'] != None:
    frameDur = 1.0 / expInfo['FrameRate']
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
if expInfo['Flickering(Hz)']:
    flickering_frameRate = round(1 / (expInfo['Flickering(Hz)'] * frameDur)) # flickering according to frame number

# Data saving
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['Participant'], expInfo['Session'], expName, expInfo['Date'])
# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
# logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

# create exp_configs
settings = ['Session', 'Participant', 'Test_Eye', 
            'Fix_Pos(deg)', 'Fix_Size(deg)', 'Fix_Color', 
            'Cursor_Color', 'Cursor_Size(deg)', 'Flickering(Hz)',  
            'Eye_Tracker']
config = {key: value for key, value in expInfo.items() if key in settings}
config.update(staircaseInfo)
config['Flickering_FrameRate'] = flickering_frameRate
config['FrameTolerance'] = frameTolerance
config['Show_Raw_Data'] = False
config['FrameDur'] = frameDur
config['FileName'] = filename
if expInfo['Eye_Tracker']:
    config['EyeTolerance'] = eyetrackingInfo['Tolerance(deg)']
else: pass

# Create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()
endExpNow = False  # flag for 'escape' or other condition => quit the exp



# -------Run experiment-------
# Reasonable values of BS position (in pix) and height when distance = 57 cm, 1 degree = 32 pix
bs_c_y = 0 # blind spot center vertical (y) coordinate value

expInfo['H_Trials'] = bs_c_y # short var name to save typing

if expInfo['Horizontal_Trials'] != 'Default':
    expInfo['H_Trials'] = eval(expInfo['Horizontal_Trials'])
if expInfo['Vertical_Trials'] != 'Default':
    expInfo['V_Trials'] = eval(expInfo['Vertical_Trials'])
ht_dots, ht_tracking = border_points(win = win, exp_config = config, path = expInfo['H_Trials'], dimension = 0)

if expInfo['Vertical_Trials'] != 'Default':
    pass
else:
    bs_c_x = np.array(ht_dots).mean(axis = 0)[0] # x coordinate of the bs center, inferred based on HT results
    expInfo['V_Trials'] = bs_c_x


vt_dots, vt_tracking = border_points(win = win, exp_config = config, path = expInfo['V_Trials'], dimension = 1)
bs_c_y = np.array(vt_dots).mean(axis = 0)[1] # x coordinate of the bs center, inferred based on HT results
bs_center = (bs_c_x, bs_c_y)
bs_height = np.array(vt_dots).max(axis = 0)[1] - np.array(vt_dots).min(axis = 0)[1]

hd_dots, hd_tracking = border_points(win = win, exp_config = config, path = bs_c_y, dimension = 0)
bs_width = np.array(hd_dots).max(axis = 0)[0] - np.array(hd_dots).min(axis = 0)[0]
bs_center_2_fixation = (bs_c_x - expInfo['Fix_Pos(deg)'][0], bs_c_y - expInfo['Fix_Pos(deg)'][1])
expInfo['BS_Center_to_Fix(deg)'] = bs_center_2_fixation
expInfo['BS_Width(deg)'] = bs_width
expInfo['BS_Height(deg)'] = bs_height

if expInfo['Staircase']:
    ori_contrast, staircase_tracking = orien_staircase(win = win, exp_config = config, ori_bs_pos = bs_center_2_fixation)
    staircaseInfo['Threshold_Orientation_Contrast'] = ori_contrast
else:
    ori_contrast = staircaseInfo['Threshold_Orientation_Contrast']

print(f'Threshold orientation contrast: {ori_contrast}\nBS width: {bs_width}\n'
      f'BS height: {bs_height}\nBS Center to Fix: {bs_center_2_fixation}')


# Initialize components for Routine "Test_complete"
test_complete = create_text(win = win, color = expInfo['Cursor_Color'], text=('Congratulations!\n\nYou have completed the whole procedure.\n\n\n\n\n\n\n'
                                  '*Please press "space" key to end...')
                            )

# run 'Test_complete'
# ------Prepare to start Routine "Test_complete"-------
# reset timers
event.clearEvents()
while True:
    test_complete.draw()
    win.flip()
    if event.getKeys(['space', 'escape']):
        break

###########################################################################


###########################################################################

# save data
expInfo['Raw_BS_Vertices(deg)'] = hd_dots + vt_dots
expInfo.update(staircaseInfo)
if expInfo['Eye_Tracker']:
    expInfo.update(eyetrackingInfo)
win.flip()

# these shouldn't be strictly necessary (should auto-save)
# thisExp.saveAsWideText(filename+'.csv')
# thisExp.saveAsPickle(filename)
exp_extraInfo = thisExp.extraInfo
data = ['ExpName', 'Date', 'Session', 'Monitor_Size(pix)', 'FrameRate', 
        'Distance(cm)', 'Participant', 'Sex', 'Age', 'Test_Eye', 'BG_Color',
        'Fix_Pos(deg)', 'Fix_Size(deg)', 'Fix_Color', 'Cursor_Color', 'Cursor_Size(deg)', 'Flickering(Hz)',
        'BS_Center_to_Fix(deg)', 'Raw_BS_Vertices(deg)', 'BS_Vertices', 'BS_Width(deg)', 'BS_Height(deg)',
        'Staircase', 'Eye_Tracker']
data += list(staircaseInfo.keys())
if expInfo['Eye_Tracker']:
    data += list(eyetrackingInfo.keys())
    try:
        ht_tracking.to_csv(filename + 'HT_tracking' + '.csv')
        vt_tracking.to_csv(filename + 'VT_tracking' + '.csv')
        hd_tracking.to_csv(filename + 'HD_tracking' + '.csv')
        staircase_tracking.to_csv(filename + 'Staircase_tracking' + '.csv')
    except Exception as e:
        print(e)
# save all raw data in a txt file, so if you want to explore the data in python,
# just read the txt file and call eval() on it, and you get a dict object.
with open(filename + '.txt', 'w', encoding = 'utf-8') as txt_file:
    txt_file.write(str(exp_extraInfo))
# some raw data in a csv file, note that data are saved as strings, so not very convenient to analyze
pd.DataFrame([exp_extraInfo]).to_csv(filename + '.csv', columns = data)
# so save numeric data in another csv file.

logging.flush()
# make sure everything is closed down
if expInfo['Eye_Tracker']:
    tracker.setRecordingState(False)
    io.quit()
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()