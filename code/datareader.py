
# coding: utf-8

# (exported from jupyter-notebook)

# # New analysis
# This version will be even more numpy-heavy. This should make it a lot easier to do multi-variate stuff.

# In[1]:

import root_numpy
import ROOT

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from collections import defaultdict
from root_numpy import root2array


# In[2]:

def scrape_regular(arr, branch):
    # generate the high-level variables
    # note that we'll probably want multiple "versions" of this function as we get more refined
    
    weights = []
    myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []
    higgspt_list = []
    id_list = []
    dphi_list = []
    maxeta_list = []
    mineta_list = []
    njets_list = []

    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]
        eventNumber = event[branch['eventNumber']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()
        higgsPT = gammaTot.Pt()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # Strategy: pick two jets which give highest Mjj
        Mjj = 0
        ibest = -1; jbest = -1
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])

                jetTot = jeti + jetj
                tempMjj = jetTot.M()

                if tempMjj > Mjj:
                    Mjj = tempMjj
                    ibest = i
                    jbest = j  

        if ibest == -1 or jbest == -1: continue
        # Mjj is ready

        Deta = abs(j0eta[ibest] - j0eta[jbest])

        leadPT = j0pT[ibest]
        subleadPT = j0pT[jbest]

        Dphi_tmp = abs(j0phi[ibest]-j0phi[jbest])
        Dphi = min(Dphi_tmp, 2*np.pi - Dphi_tmp)
        
        maxeta = max(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        mineta = min(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        
        njets_real = len(j0pT) - sum(j0_isTightPhoton)
        
        myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        higgspt_list.append(higgsPT)
        weights.append(eventWeight)
        id_list.append(eventNumber)
        dphi_list.append(Dphi)
        maxeta_list.append(maxeta)
        mineta_list.append(mineta)
        njets_list.append(njets_real)
        
    return {'weight':weights,
            'myy':myy_list,
            'mjj':mjj_list,
            'deta':deta_list, 
            'leadpt':leadpt_list,
            'subleadpt':subleadpt_list,
            'higgspt':higgspt_list,
            'id':id_list,
            'dphi':dphi_list,
            'maxeta':maxeta_list,
            'mineta':mineta_list,
            'njets':njets_list}

# In[3]:

def scrape_folder(folder, branch, scraper, maxfiles=2):
    allVars = defaultdict(list)
    
    for i, fname in enumerate(sorted(os.listdir(folder))):
        if i >= maxfiles: break # to go fast
        
        filepath = os.path.join(folder, fname)
        print('reading file', filepath)
        arr = root2array(filepath, 'Nominal')
        newVars = scraper(arr, branch)
        for var,var_list in newVars.items():
            allVars[var] += var_list

    # normalize weight
    if 'weight' in allVars:
        allVars['weight'] = np.array(allVars['weight']) / np.sum(allVars['weight'])
            
    return allVars


# In[4]:

# Functions to use real numpy histograms

def histograms(var_name, bin_range, sg_data, bg_data, nbins=100):
    # Make signal and background histograms for a given high-level variable
    
    h_sg, bin_edges_sg = np.histogram(sg_data[var_name], bins=nbins, range=bin_range, weights=sg_data['weight'])
    h_bg, bin_edges_bg = np.histogram(bg_data[var_name], bins=nbins, range=bin_range, weights=bg_data['weight'])
    assert (bin_edges_sg == bin_edges_bg).all()
    
    return h_sg, h_bg, bin_edges_sg

def plot_histograms(h_sg, h_bg, bin_edges, xlabel=None, savefig=None):
    # Plot signal and background histograms
    
    fig = plt.figure(figsize=(6,4))
    plt.step(bin_edges[:-1], h_sg, color='teal', linewidth=1, label='VBF')
    plt.step(bin_edges[:-1], h_bg, color='darkred', linewidth=1, label='ggF')
    plt.yscale('log', nonposy='clip')
    if xlabel:
        plt.xlabel(xlabel)

    plt.legend()

    if savefig:
        if savefig[:8] == '../imgs/':
            plt.savefig(savefig)
        else:
            print('Warning: not saving image (please save to ../imgs/)')
        
    plt.show()

def calc_ROC(signal_hist, background_hist):
    # return list of x_arr and y_arr to plot
    nbins = len(signal_hist)
    assert nbins == len(background_hist)
    
    x_list = []
    y_list = []
    
    for i in range(nbins):
        true_pos = np.sum(signal_hist[i:])
        false_pos = np.sum(background_hist[i:])
        
        x_list.append(true_pos)
        y_list.append(false_pos)
        
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    
    return x_arr, y_arr
    
def draw_ROC(signal_hist, background_hist, color=None, lw=None, show=True, title=None, label=None, preshow=(lambda: None), savefig=None):
    x_arr, y_arr = calc_ROC(signal_hist, background_hist)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.plot(x_arr, y_arr, color=color, lw=lw, label=label)
    plt.plot([0,1],[0,1], '--', color='bisque') # straight line
    
    xs = np.linspace(0,1,100)
    plt.plot(xs, xs**2, '--', color='gray') # signal / sqrt(background) line
    
    if title:
        plt.title(title)
    
    if show and label:
        plt.legend()
    
    preshow()

    if savefig:
        if savefig[:8] == '../imgs/':
            plt.savefig(savefig)
        else:
            print('Warning: not saving image (please save to ../imgs/)')
    
    if show:
        plt.show()

 
    
# def ROC(signal_hist, background_hist, verbose=True, title='ROC', filename=None):
#     # Draws ROC curve, returns AUC
#     x_arr, y_arr = calc_ROC(signal_hist, background_hist)

    
    
#     # AUC
#     dx_arr = x_arr[:-1] - x_arr[1:]
#     auc = dx_arr.dot(y_arr[:-1] + y_arr[1:])/2

#     if verbose:
#         print('AUC', auc)
    
#     if verbose or filename:
#         draw_ROC(x_arr, y_arr, show=False)
#         plt.title(title)
#         if filename:
#             plt.savefig(filename)
#         if verbose:
#             plt.show()
#         else:
#             plt.clr()
    
#     return auc

# Use Significance
# (N-B)/(sqrt B) = S/sqrt(B)
# show line of random guessing
# show line E_s = sqrt(E_B)



# ## Now we do it again using hacky $M_{jj}$

# In[26]:

def scrape_hacky(arr, branch):
    # select two jets based on M_jj - 2*gammaTot
    
    weights = []
    #myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []
    njets_list = []

    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # Strategy: pick two jets which give highest "recoil Mjj"
        recoilMjj = 0
        jet0 = None; jet1 = None
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])
    

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])
                
                jetTot = jeti - gammaTot + jetj - gammaTot
                tempMjj = jetTot.M()

                if tempMjj > recoilMjj:
                    recoilMjj = tempMjj
                    jet0 = jeti
                    jet1 = jetj

        if jet0 == None or jet1 == None: continue
        Mjj = (jet0 + jet1).M()
        # Mjj is ready
        
        
        Deta = abs(jet0.Eta() - jet1.Eta())

        leadPT = max(jet0.Pt(), jet1.Pt())
        subleadPT = min(jet0.Pt(), jet1.Pt())

        njets_real = len(j0pT) - sum(j0_isTightPhoton)
        
        #myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        weights.append(eventWeight)
        njets_list.append(njets_real)

    return {'weight':weights,
            #'myy':myy_list,
            'mjj':mjj_list,
            'deta':deta_list, 
            'leadpt':leadpt_list,
            'subleadpt':subleadpt_list,
            'njets':njets_list}

def scrape_selective_gen(multiple):
    return lambda arr, branch: scrape_selective_helper(arr, branch, multiple)
    

def scrape_selective_helper(arr, branch, multiple):
    # regular scrape, but only applies when Mjj is greater than gammaTot (or something like that)
    
    weights = []
    #myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []

    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # run this first loop just as weed-out
        
        pseudoMjj = 0
        jet0 = None; jet1 = None
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])
    

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])
                
                jetTot = jeti + jetj - gammaTot*2
                tempMjj = jetTot.M()

                if tempMjj > pseudoMjj:
                    pseudoMjj = tempMjj
                    jet0 = jeti
                    jet1 = jetj

        if jet0 == None or jet1 == None: continue

        pseudoMjj = 0
        jet0 = None; jet1 = None
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])
    

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])
                
                jetTot = jeti + jetj
                tempMjj = jetTot.M()

                if tempMjj > pseudoMjj:
                    pseudoMjj = tempMjj
                    jet0 = jeti
                    jet1 = jetj

        if jet0 == None or jet1 == None: continue
        Mjj = (jet0 + jet1).M()
        # Mjj is ready
        
        
        Deta = abs(jet0.Eta() - jet1.Eta())

        leadPT = max(jet0.Pt(), jet1.Pt())
        subleadPT = min(jet0.Pt(), jet1.Pt())

        #myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        weights.append(eventWeight)

    return {'weight':weights,
             #'myy':myy_list,
             'mjj':mjj_list,
             'deta':deta_list, 
             'leadpt':leadpt_list,
             'subleadpt':subleadpt_list}




def scrape_higgs(arr, branch):
    # do everything in Higgs frame
    
    weights = []
    #myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []

    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # Strategy: pick two jets which give highest "recoil Mjj"
        higgsMjj = 0
        jet0 = None; jet1 = None
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])
            jeti.Boost(-gammaTot.BoostVector())

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])
                jetj.Boost(-gammaTot.BoostVector())
                
                jetTot = jeti + jetj
                tempMjj = jetTot.M()

                if tempMjj > higgsMjj:
                    higgsMjj = tempMjj
                    jet0 = jeti
                    jet1 = jetj

        if jet0 == None or jet1 == None: continue

        
        Mjj = (jet0 + jet1).M()
        # Mjj is ready
        
        Deta = abs(jet0.Eta() - jet1.Eta())

        leadPT = max(jet0.Pt(), jet1.Pt())
        subleadPT = min(jet0.Pt(), jet1.Pt())

        #myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        weights.append(eventWeight)

    return {'weight':weights,
             #'myy':myy_list,
             'mjj':mjj_list,
             'deta':deta_list, 
             'leadpt':leadpt_list,
             'subleadpt':subleadpt_list}


def scrape_maxpt(arr, branch):
    # "greedily" use the two jets with the highest pT
    
    weights = []
    myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []
    subsubleadpt_list = []
    higgspt_list = []
    id_list = []
    dphi_list = []
    maxeta_list = []
    mineta_list = []
    njets_list = []
    dr12_list = []
    dr13_list = []
    dr23_list = []
    mindr_list = []
    eta3_list = []
    
    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]
        eventNumber = event[branch['eventNumber']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()
        higgsPT = gammaTot.Pt()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # Strategy: pick two non-tight-photon jets with highest pT
        # Note that this code works because the jets are already ordered by pT
        ibest = -1; jbest = -1; kbest = -1 # (kbest for subsubleading jet)
        for i in range(njets):
            if j0_isTightPhoton[i]: continue

            if ibest == -1:
                ibest = i
            elif jbest == -1:
                jbest = i
            elif kbest == -1:
                kbest = i
                break

        if ibest == -1 or jbest == -1: continue
        # two jets have been chosen

        # calculate Mjj
        jeti = ROOT.TLorentzVector()
        jeti.SetPtEtaPhiM(j0pT[ibest], j0eta[ibest], j0phi[ibest], j0m[ibest])

        jetj = ROOT.TLorentzVector()
        jetj.SetPtEtaPhiM(j0pT[jbest], j0eta[jbest], j0phi[jbest], j0m[jbest])
        
        Mjj = (jeti + jetj).M()

        # Calculate other high-level variables
        Deta = abs(j0eta[ibest] - j0eta[jbest])

        leadPT = j0pT[ibest]
        subleadPT = j0pT[jbest]
        subsubleadPT = j0pT[kbest] if kbest != -1 else 0

        Dphi_tmp = abs(j0phi[ibest]-j0phi[jbest])
        Dphi = min(Dphi_tmp, 2*np.pi - Dphi_tmp)
        
        maxeta = max(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        mineta = min(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        
        njets_real = len(j0pT) - sum(j0_isTightPhoton)


        # if at least three jets, calculate dr between them:
        if kbest != -1:
            dr12 = dr_etas_phis(j0eta[ibest], j0eta[jbest], j0phi[ibest], j0phi[jbest])
            dr13 = dr_etas_phis(j0eta[ibest], j0eta[kbest], j0phi[ibest], j0phi[kbest])
            dr23 = dr_etas_phis(j0eta[jbest], j0eta[kbest], j0phi[jbest], j0phi[kbest])
            mindr = min(dr12, dr13, dr23)
        else:
            # if only 2 jets, don't do anything with these variables (just default them to 0)
            dr12 = 0; dr13 = 0; dr23 = 0
            mindr = 0

        eta3 = np.abs(j0eta[kbest]) if kbest != -1 else 0
        
        myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        subsubleadpt_list.append(subsubleadPT)
        higgspt_list.append(higgsPT)
        weights.append(eventWeight)
        id_list.append(eventNumber)
        dphi_list.append(Dphi)
        maxeta_list.append(maxeta)
        mineta_list.append(mineta)
        njets_list.append(njets_real)
        dr12_list.append(dr12)
        dr13_list.append(dr13)
        dr23_list.append(dr23)
        mindr_list.append(mindr)
        eta3_list.append(eta3)
        
    return {'weight':weights,
            'myy':myy_list,
            'mjj':mjj_list,
            'deta':deta_list, 
            'leadpt':leadpt_list,
            'subleadpt':subleadpt_list,
            'subsubleadpt':subsubleadpt_list,
            'higgspt':higgspt_list,
            'id':id_list,
            'dphi':dphi_list,
            'maxeta':maxeta_list,
            'mineta':mineta_list,
            'njets':njets_list,
            'dr12':dr12_list,
            'dr13':dr13_list,
            'dr23':dr23_list,
            'mindr':mindr_list,
            'eta3':eta3_list}



def dr_etas_phis(eta1, eta2, phi1, phi2):
    # calculating dr
    deta = abs(eta1 - eta2)
    raw_dphi = abs(phi1 - phi2)
    dphi = min(raw_dphi, 2*np.pi - raw_dphi)
    return np.sqrt(deta**2 + dphi**2)



def scrape_maxdeta(arr, branch):
    # Pick the two jets to maximize deta
    
    weights = []
    myy_list = []
    mjj_list = []
    deta_list = []
    leadpt_list = []
    subleadpt_list = []
    higgspt_list = []
    id_list = []
    dphi_list = []
    maxeta_list = []
    mineta_list = []
    njets_list = []

    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]
        eventNumber = event[branch['eventNumber']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()
        higgsPT = gammaTot.Pt()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        njets = len(j0pT)

        # Strategy: pick two non-tight-photon jets with biggest deta separation
        ibest = -1; jbest = -1
        deta = 0
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            
            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue

                tempDeta = abs(j0eta[i] - j0eta[j])

                if tempDeta > deta:
                    ibest = i
                    jbest = j
                
        if ibest == -1 or jbest == -1: continue
        # two jets have been chosen

        # calculate Mjj
        jeti = ROOT.TLorentzVector()
        jeti.SetPtEtaPhiM(j0pT[ibest], j0eta[ibest], j0phi[ibest], j0m[ibest])

        jetj = ROOT.TLorentzVector()
        jetj.SetPtEtaPhiM(j0pT[jbest], j0eta[jbest], j0phi[jbest], j0m[jbest])
        
        Mjj = (jeti + jetj).M()

        # Calculate other high-level variables
        Deta = abs(j0eta[ibest] - j0eta[jbest])

        leadPT = j0pT[ibest]
        subleadPT = j0pT[jbest]

        Dphi_tmp = abs(j0phi[ibest]-j0phi[jbest])
        Dphi = min(Dphi_tmp, 2*np.pi - Dphi_tmp)
        
        maxeta = max(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        mineta = min(np.abs(j0eta[ibest]), np.abs(j0eta[jbest]))
        
        njets_real = len(j0pT) - sum(j0_isTightPhoton)
        
        myy_list.append(Myy)
        mjj_list.append(Mjj)
        deta_list.append(Deta)
        leadpt_list.append(leadPT)
        subleadpt_list.append(subleadPT)
        higgspt_list.append(higgsPT)
        weights.append(eventWeight)
        id_list.append(eventNumber)
        dphi_list.append(Dphi)
        maxeta_list.append(maxeta)
        mineta_list.append(mineta)
        njets_list.append(njets_real)
        
    return {'weight':weights,
            'myy':myy_list,
            'mjj':mjj_list,
            'deta':deta_list, 
            'leadpt':leadpt_list,
            'subleadpt':subleadpt_list,
            'higgspt':higgspt_list,
            'id':id_list,
            'dphi':dphi_list,
            'maxeta':maxeta_list,
            'mineta':mineta_list,
            'njets':njets_list}


def scrape_low(arr, branch):
    # let's try dumping the low-level parameters
    
    weights = []
    njets_list = []
    y0pt_list = []
    y0eta_list = []
    y0phi_list = []
    y0m_list = []
    y1pt_list = []
    y1eta_list = []
    y1phi_list = []
    y1m_list = []
    j0pt_list = []
    j0eta_list = []
    j0phi_list = []
    j0m_list = []
    j1pt_list = []
    j1eta_list = []
    j1phi_list = []
    j1m_list = []

    
    for jentry, event in enumerate(arr):
        gammapT = event[branch['gammapT']]
        gammaeta = event[branch['gammaeta']]
        gammaphi = event[branch['gammaphi']]
        gammam = event[branch['gammam']]
        gamma_isTight = event[branch['gamma_isTight']]
        j0pT = event[branch['j0pT']]
        j0eta = event[branch['j0eta']]
        j0phi = event[branch['j0phi']]
        j0m = event[branch['j0m']]
        j0_isTightPhoton = event[branch['j0_isTightPhoton']]
        j0_fJVT_Tight = event[branch['j0_fJVT_Tight']]
        eventWeight = event[branch['eventWeight']]

        ### Event-wide Myy cut
        if len(gammapT) != 2: continue
        # only tight photons
        if not gamma_isTight[0] or not gamma_isTight[1]: continue

        gamma0 = ROOT.TLorentzVector()
        gamma0.SetPtEtaPhiM(gammapT[0], gammaeta[0], gammaphi[0], gammam[0])
        gamma1 = ROOT.TLorentzVector()
        gamma1.SetPtEtaPhiM(gammapT[1], gammaeta[1], gammaphi[1], gammam[1])

        gammaTot = gamma0 + gamma1
        Myy = gammaTot.M()

        ypt0 = gammapT[0]
        ypt1 = gammapT[1]
        if max(ypt0, ypt1) < .35*Myy or max(ypt0, ypt1) < .25*Myy: continue
        ### end Event-wide Myy cut

        
        ## next, we look for the two "recoil" jets
        njets = len(j0pT)
        Mjj = 0
        jet0 = None; jet1 = None
        for i in range(njets):
            if j0_isTightPhoton[i]: continue
            # if j0_fJVT_Tight[i]: continue

            jeti = ROOT.TLorentzVector()
            jeti.SetPtEtaPhiM(j0pT[i], j0eta[i], j0phi[i], j0m[i])
    

            for j in range(i+1, njets):
                if j0_isTightPhoton[j]: continue
                # if j0_fJVT_Tight[j]: continue            

                jetj = ROOT.TLorentzVector()
                jetj.SetPtEtaPhiM(j0pT[j], j0eta[j], j0phi[j], j0m[j])
                
                jetTot = jeti + jetj # let's do this properly
                tempMjj = jetTot.M()

                if tempMjj > Mjj:
                    Mjj = tempMjj
                    jet0 = jeti
                    jet1 = jetj

        if jet0 == None or jet1 == None: continue
        # Now we've picked our jets, the same way as in `scrape_regular`
        
        # All that remains is to fill out our low-level parameters
        
        weights.append(eventWeight)
        njets_list.append(len(j0pT) - sum(j0_isTightPhoton))
        y0pt_list.append(gammapT[0])
        y0eta_list.append(gammaeta[0])
        y0phi_list.append(gammaphi[0])        
        y0m_list.append(gammam[0])
        y1pt_list.append(gammapT[1])
        y1eta_list.append(gammaeta[1])
        y1phi_list.append(gammaphi[1])        
        y1m_list.append(gammam[1])
        j0pt_list.append(jet0.Pt())
        j0eta_list.append(jet0.Eta())
        j0phi_list.append(jet0.Phi())
        j0m_list.append(jet0.M())
        j1pt_list.append(jet1.Pt())
        j1eta_list.append(jet1.Eta())
        j1phi_list.append(jet1.Phi())
        j1m_list.append(jet1.M())

    return {'weight':weights,
            'njets':njets_list,
            'y0pt':y0pt_list,
            'y0eta':y0eta_list,
            'y0phi':y0phi_list,
            'y0m':y0m_list,
            'y1pt':y1pt_list,
            'y1eta':y1eta_list,
            'y1phi':y1phi_list,
            'y1m':y1m_list,
            'j0pt':j0pt_list,
            'j0eta':j0eta_list,
            'j0phi':j0phi_list,
            'j0m':j0m_list,
            'j1pt':j1pt_list,
            'j1eta':j1eta_list,
            'j1phi':j1phi_list,
            'j1m':j1m_list}



###

# Now we'll add a visualization program

def event3d(arr, branch, eventidx):
    print('Event display for event idx', eventidx)

    event = arr[eventidx]

    j0_isTightPhoton = event[branch['j0_isTightPhoton']]
    j0pT = event[branch['j0pT']]
    j0eta = event[branch['j0eta']]
    j0phi = event[branch['j0phi']]
    j0m = event[branch['j0m']]
    gammapT = event[branch['gammapT']]
    gammaeta = event[branch['gammaeta']]
    gammaphi = event[branch['gammaphi']]
    gammam = event[branch['gammam']]
    

    ROOT.TEveManager.Create()

    # Make some circles around  $\abs{\eta} = 2.5$

    ps = ROOT.TEvePointSet("ps")
    scale = 100 # make the circle bigger by this amount
    for i in np.linspace(0, 2*np.pi, 20):
        vec = getTEveVector(2.5, i)
        ps.SetNextPoint(scale * vec.fX, scale * vec.fY, scale * vec.fZ)
        ps.SetNextPoint(scale * vec.fX, scale * vec.fY, -scale * vec.fZ)
    ps.SetNextPoint(0,0,0) # origin
        
    ps.SetMarkerSize(2)
    ps.SetMarkerStyle(4)
    ps.SetMarkerColor(5)
    
    # etaCone = ROOT.TEveJetCone("etaCone")
    # etaCone.SetPickable(True)
    # # etaCone.SetCylinder(100,100)
    # etaCone.SetRadius(50)
    # etaCone.AddCone(5,1,1)
    

    
    zAxis = ROOT.TEveStraightLineSet("zAxis")
    zAxis.SetLineColor(ROOT.kGray)
    zAxis.SetLineWidth(2)
    zAxis.AddLine(0,0,-300, 0,0,300)
    
    eveJets = ROOT.TEveStraightLineSet("eveJets")
    eveJets.SetLineColor(ROOT.kRed)
    eveJets.SetLineWidth(2)
    #
    for i, (pt, eta, phi, m) in enumerate(zip(j0pT, j0eta, j0phi, j0m)):
        # skip "tight photon" jets
        if j0_isTightPhoton[i]: continue

        # using m_T^2 = m^2 + p_T^2
        E = np.sqrt(m**2 + pt**2) * np.cosh(eta)
        vector = getTEveVector(eta, phi) * E
        eveJets.AddLine(0,0,0, vector.fX, vector.fY, vector.fZ)

        print('biggest eta difference:', max(j0eta) - min(j0eta))

    gammas = ROOT.TEveStraightLineSet("gammas")
    gammas.SetLineColor(ROOT.kGreen)
    gammas.SetLineWidth(2)
    #
    for pt, eta, phi, m in zip(gammapT, gammaeta, gammaphi, gammam):
        E = np.sqrt(m**2 + pt**2) * np.cosh(eta)
        vector = getTEveVector(eta, phi) * E
        gammas.AddLine(0,0,0, vector.fX, vector.fY, vector.fZ)
    

    ROOT.gEve.AddElement(ps)
    ROOT.gEve.AddElement(zAxis)
    ROOT.gEve.AddElement(eveJets)
    ROOT.gEve.AddElement(gammas)

    ROOT.gEve.Redraw3D(ROOT.kTRUE)
        
    input('waiting...')

    
# def print_eventids(arr, branch):
#     for jentry, event in enumerate(arr):
#         if jentry > 100:
#             break

#         print('jentry', jentry)
#         eventNumber = event[branch['eventNumber']]
#         print(eventNumber)


def getTEveVector(eta, phi):
    tm = ROOT.TMath
    vec = ROOT.TEveVector(tm.Cos(phi) / tm.CosH(eta),
                          tm.Sin(phi) / tm.CosH(eta),
                          tm.TanH(eta))
    return vec


def check_event(arr, branch, nr, njets=3):
    event = arr[nr]

    gammapT = event[branch['gammapT']]
    j0_isTightPhoton = event[branch['j0_isTightPhoton']]

    # want exactly two photons
    if len(gammapT) != 2:
        return False
    
    # there need to be at least two non-tight jets
    if len(j0_isTightPhoton) - sum(j0_isTightPhoton) < 2:
        return False

    # njet "cut"?
    if njets > 0:
        if len(j0_isTightPhoton) - sum(j0_isTightPhoton) != njets:
            return False
        
	
    return True


if __name__ == '__main__':
    signal = False
    myrange = 20
    njets = 3

    # signal = False
    # myrange = range(0,25)
    
    print('signal?', signal)
    print('njets?', njets)
    
    if signal:
        filename = '../data/vbfroot/data-CxAOD-0.root'
    else:
        filename = '../data/ggfroot/data-CxAOD-0.root'
        
    branch_names = root_numpy.list_branches(filename, 'Nominal')
    branch = {name:number for number,name in enumerate(branch_names)}
    
    # vbf_reg = scrape_folder('../data/vbfroot/', branch, scrape_regular, maxfiles=1)
    arr = root2array(filename, 'Nominal')

    mycount = 0
    for i in range(len(arr)):
        if check_event(arr, branch, i, njets):
            print(i, 'weight', arr[i][branch['eventWeight']])
            event3d(arr, branch, i)
            mycount += 1

        if mycount == myrange:
            break


    

