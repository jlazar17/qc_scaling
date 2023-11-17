import numpy as np
import random
import itertools

from time import time
from tqdm import tqdm
from typing import Union

class OpString(str):
    
    def __init__(self, s):
        self.s = s
        self.n = None
        self._complementer = None
        
    def __add__(self, other):
        if len(self)!=len(other):
            raise ValueError
        s = ''
        for i in range(len(self)):
            s += str((int(self.s[i])+int(other.s[i]))%3)
        return OpString(s)
            
    def __sub__(self, other):
        if len(self)!=len(other):
            raise ValueError
        s = ''
        for i in range(len(self)):
            s += str((int(self.s[i])-int(other.s[i]))%3)
        return OpString(s)
    
    def op_n(self):
        if self.n is None:
            n = 0
            for i, x in enumerate(reversed(self.s)):
                n += int(x)*3**i
            self.n = n
        return self.n
    
    def complement(self):
        if self._complementer is None:
            self._complementer = OpString("3"*len(self.s))
        return self._complementer - self

def convert_basis(n: Union[int, str], inbase: int, outbase: int) -> Union[str, int]:
    intn = int(n)
    if intn == 0:
        return '0'
    if inbase != 10:
        decimal_number = 0
        for n, x in enumerate(reversed(str(n))):
            decimal_number += int(x) * np.power(inbase, n)
    else:
        decimal_number = intn

    remainder_stack = []

    while decimal_number > 0:
        remainder  = decimal_number % outbase
        remainder_stack.append(remainder)
        decimal_number = decimal_number // outbase
    return ''.join([str(x) for x in remainder_stack[::-1]])

nqubit=14
oplist=[OpString(''.join(tritstrings)) for tritstrings in list(itertools.product(*([['0','2','1']]*nqubit)))*2]
#Run this just once
oplist_short = [OpString(''.join(tritstrings)) for tritstrings in list(itertools.product(*([['0','2','1']]*nqubit)))]
op_to_index_dict = dict([[oplist_short[i],i] for i in range(len(oplist_short))])
#Loading fingerprint files. Section 5 computes these files
save_folder = "/n/home12/jlazar/qc_scaling/scaling_C_memories"
loaded_stringfingerprint_even = np.load(f'{save_folder}/stringfingerprint_even.npy')
loaded_stringfingerprint_odd = np.load(f'{save_folder}/stringfingerprint_odd.npy')
loaded_first_column = np.load(f'{save_folder}/stringfingerprint_firstcolumn.npy')
loaded_patterns64x64 = np.load(f'{save_folder}/stringfingerprint_patterns64x64.npy')
stringfingerprint_even = loaded_stringfingerprint_even
stringfingerprint_odd = loaded_stringfingerprint_odd
first_column = loaded_first_column
patterns64x64 = loaded_patterns64x64
alphalist=[('0'*nqubit+bin(i).replace("0b", ""))[-nqubit:] for i in range(2**nqubit)]
#Precalculation common to every state and context. Run just once.
preadd=[OpString(('0'*nqubit+convert_basis(str(i),10,2))[-nqubit:]) for i in range(2**nqubit)]
preop_add_even = list(filter(lambda thang: sum([int(i) for i in list(thang)])%2==0, preadd))
preop_add_odd = list(filter(lambda thang: sum([int(i) for i in list(thang)])%2==1, preadd))
op_add_even=[OpString('0'*nqubit)]+[OpString('1'*nqubit)+i for i in preop_add_even]
op_add_odd=[OpString('0'*nqubit)]+[OpString('1'*nqubit)+i for i in preop_add_odd]
random.seed(100)
#Using an arbitrary seed, for consistency, and to guarantee that we aren't cheating by generating the data
#to match some quantum states statistics.
randombitstring = [int(i) for i in format(random.getrandbits(int((3**nqubit-1)/2)),'0b')]
zeropad = list(np.zeros(int((3**nqubit-1)/2), dtype=int))
data = (zeropad+randombitstring)[-int((3**nqubit-1)/2):]
samplealpha = '0'*nqubit

randombitstring2 = [int(i) for i in format(random.getrandbits(int((3**nqubit-1)/2)),'0b')]
pair_flipper = (zeropad+randombitstring2)[-int((3**nqubit-1)/2):]
pair_parities = [(pairbit if pair_flipper[i]==0 else [1-pairbit[0],1-pairbit[1]]) for i,pairbit in enumerate([([0,0] if log_bit==0 else [0,1]) for log_bit in data])]+[[0]]
target_parities = [pair_parities[i//2][i%2] for i in range(3**nqubit)]
ctx_alpha = [(i,str(i//(3**nqubit))+samplealpha,OpString(''.join(tritstrings))) for i,tritstrings in enumerate(list(itertools.product(*([['0','2','1']]*nqubit)))*2)]
#This block uses ~1000 states.
randomlyplacedones = [(list(np.zeros(x-1, dtype=int))+[1]+list(np.zeros(10000-x, dtype=int))) for x in [random.randint(1, 10000) for i in range(956)]]
randomlyplacedones2 = []
for arr in randomlyplacedones:
    randomlyplacedones2 += arr
ctx_onoff = np.array(randomlyplacedones2+list(np.zeros(5938, dtype=int)))
which_ctx_ON = np.array([i for i in range(len(ctx_onoff))])[ctx_onoff==1]
which_ctx_ONv2 = [[int((('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-(nqubit+1):])[0]),OpString(('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-nqubit:]).complement()] for ctx_ind in which_ctx_ON]
which_alpha_on = [ctx_alpha[i] for i in which_ctx_ON]
which_alpha_on_preham = [[i,0] for i in which_alpha_on]
previous_stateselection_list = [i for i in np.load(f"{save_folder}/scored_states_nofall_multi_smartpick_ith.npy",allow_pickle=True)]

def parityanalytic(XXXpiece, beta2):
    beta = XXXpiece[2]
    thetaS = int(XXXpiece[1][0])#integer (0 or 1)
    thetaZ = int(XXXpiece[1][1])#integer (0 or 1)
    alpha = [int(i) for i in (XXXpiece[1][2:]+'0')]#integer vector (0 or 1)
    betap = beta-beta2 #opstring
    J = [(1-(int(digit) - 1)) % 2 for digit in betap]
    if all(element == '0' for element in betap):
        parity=sum(alpha) % 2
    elif (any(element == '0' for element in betap)):
        parity='undefined'
    elif ((sum(J)+thetaS)%2==1):
        parity='undefined'
    else:
        parity=(thetaZ+int((sum(J)+thetaS)/2)+np.dot(J,alpha))%2

    return parity

def mean_z(things):
    return np.mean(things) if len(things)>0 else 0.5
#We define this, to cover the case of zero-length lists

def mean_zz(things):
    return np.mean(things) if len(things)>0 else None
#We define this, to cover the case of zero-length lists. However, we leave a None if that's the case.

def round_z(number):
    if number==0.5 or number==None:
        return number
    return round(number)

def get_context(XXXterm,op_add_even=op_add_even,op_add_odd=op_add_odd):
    sbit=int(XXXterm[1][0])
    op_gen_to_measure=XXXterm[2]
    if sbit == 0:
        op_context=[op_gen_to_measure+i for i in op_add_even]
    else:
        op_context=[op_gen_to_measure+i for i in op_add_odd]
    return op_context

def score_states(
    pre_scored_states: List[Tuple[State, int]],
    data=data: List[bool],
    nqubit: int = nqubit,
    op_to_index_dict: Dict[ParityOperator, int] = op_to_index_dict,
    oplist=oplist
):

    XXXlist0 = [state[0] for state in pre_scored_states]

    # This is a list of list of ints
    # Each entry contains the indices <-> operators for a context
    contextlist0 = [[op_to_index_dict[op] for op in get_context(XXX)] for XXX in XXXlist0]
    # We're now preallocating an array the will count how many times a specific operator is seen
    PO_count=np.zeros(3**nqubit)
    # Here we fill that array
    for contextindexes in contextlist0:
        for ii in contextindexes:
            PO_count[ii]+=1

    # Find if neighbors both can have defined values
    couples_appear = np.array([(PO_count[i*2]>0 and PO_count[i*2+1]>0) for i in range(int((3**nqubit-1)/2))])
    # And get the indices for which sets of neighbors have a defined value
    which_couples_appear = np.arange(int((3**nqubit-1)/2))[couples_appear]

    # boolean mask for whether an operator appears with it's couple
    #Indicate whether each PO appears with its couple
    which_POs_appear_wcouple = np.array([None]*(2*len(which_couples_appear)))
    for i in range(len(which_couples_appear)):
        which_POs_appear_wcouple[2*i]=which_couples_appear[i]*2
        which_POs_appear_wcouple[2*i+1]=which_couples_appear[i]*2+1

    POs_appear_wcouple = np.full(3**nqubit, False)
    POs_appear_wcouple[which_couples_appear] = True
    #POs_appear_wcouple = np.zeros(3**nqubit,dtype=bool)
    #for i in which_POs_appear_wcouple:
    #    POs_appear_wcouple[i] = True

    # For each state in the selection: indicate which context POs appear with their couple (does not depend on alpha)
    # Remember contextlist0 is a list of lists of indices
    # Each sub-list is a context
    whichcontextPOs_appear_wcouple = [None] * len(contextlist0)
    for istate, context in enumerate(contextlist0):
        # This 
        does_it_have_couple = np.array([POs_appear_wcouple[i] for i in context], dtype=bool)
        whichcontextPOs_appear_wcouple[istate] = np.array(context)[does_it_have_couple]
    ################

    #Code dependent on the alphas
    ################
    #For each state in the selection: Calculate the parities of the POs which appear with their couple
    parities_wcouple = [None]*len(contextlist0)
    for istate,context in enumerate(contextlist0):
        parities_wcouple[istate] = [(PO, parityanalytic(XXXlist0[istate], oplist[PO])) for PO in whichcontextPOs_appear_wcouple[istate]]

    #Calculate the majority parities of the POs that appear with their couples
    global_parities=[[]]*(3**nqubit)
    for state_stats in parities_wcouple:
        for state_stat in state_stats:
            global_parities[state_stat[0]]=global_parities[state_stat[0]]+[state_stat[1]]

    # Score each state based on the number of POs that appear with their couple with a parity favorable to the data,
    # using the current majority of the PO couple.
    state_score=[None]*len(contextlist0)
    for istate, which_appear in enumerate(whichcontextPOs_appear_wcouple):
        score = 0
        for iappear, PO in enumerate(which_appear):
            if PO % 2==0:
                complement_mean_parity = round_z(mean_zz(global_parities[PO+1]))
                databiti = data[int(PO/2)]
            else:
                complement_mean_parity = round_z(mean_zz(global_parities[PO-1]))
                databit = data[int((PO-1)/2)]

            stateparity = parities_wcouple[istate][iappear][1]
            if databit == 0:
                if stateparity == complement_mean_parity:
                    score += 1
                elif stateparity == (1-complement_mean_parity):
                    score -= 1
            else:
                if stateparity == (1-complement_mean_parity):
                    score += 1
                elif stateparity == complement_mean_parity:
                    score -= 1
        state_score[istate] = score

    ################

    scored_states=[[XXXlist0[istate],score] for istate,score in enumerate(state_score)]
    return scored_states

def intersect_set(a, b):
        return sorted(set(a) & set(b))

#NON SORTED CONTEXT. It appears like sorting the context messes up calculating the parity with fingerprint.
#We now opt to NOT sort the context.
def precalculate(
    scored_states,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64
):
    ################################################################################################################
    # List of States
    XXXlist0 = [state[0] for state in scored_states]

    # List of lists
    # each list contains indices of context generated by corresponding state
    contextlist0 = [[op_to_index_dict[op] for op in get_context(XXX)] for XXX in XXXlist0]

    #Count number of times each PO appears, and whether each couple (both POs) appears.
    PO_count = np.zeros(3**nqubit)
    for contextindexes in contextlist0:
        for ii in contextindexes:
            PO_count[ii]+=1
    couples_appear = np.array([(PO_count[i*2]>0 and PO_count[i*2+1]>0) for i in range(int((3**nqubit-1)/2))])
    which_couples_appear = np.arange(int((3**nqubit-1)/2))[couples_appear]

    #Indicate whether each PO appears with its couple
    which_POs_appear_wcouple=np.array([None]*(2*len(which_couples_appear)))
    for i in range(len(which_couples_appear)):
        which_POs_appear_wcouple[2*i]=which_couples_appear[i]*2
    for i in range(len(which_couples_appear)):
        which_POs_appear_wcouple[2*i+1]=which_couples_appear[i]*2+1
    #Same, but with a 3**n True/False list
    POs_appear_wcouple=np.zeros(3**nqubit,dtype=bool)
    for i in which_POs_appear_wcouple:
        POs_appear_wcouple[i]=True

    #For each state in the selection: indicate which context POs appear with their couple (does not depend on alpha)
    whichcontextPOs_appear_wcouple = [None]*len(contextlist0)
    contextPOs_appear_wcouple = [None]*len(contextlist0)
    for istate,context in enumerate(contextlist0):
        does_it_have_couple = np.array([POs_appear_wcouple[i] for i in context],dtype=bool)
        contextPOs_appear_wcouple[istate] = does_it_have_couple
        whichcontextPOs_appear_wcouple[istate] = np.array(context)[does_it_have_couple]
    ################

    #For each state, this is the position in the context where the POs with appearing couples are.
    position_contextPOs_wcouple=[np.arange(len(contextlist0[0]))[appearwcouple] for appearwcouple in contextPOs_appear_wcouple]
    #We should be able to build a target parity of the size of the context using this.

    ################################################################################################################


    precalculations=[None]*6
    precalculations[0] = contextlist0
    precalculations[1] = PO_count
    precalculations[2] = which_POs_appear_wcouple
    precalculations[3] = POs_appear_wcouple
    precalculations[4] = whichcontextPOs_appear_wcouple
    precalculations[5] = position_contextPOs_wcouple
    return precalculations


def perturb_precalculate(
    scored_states,
    new_state,
    precalculations0,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64
):

    old_contextlist = precalculations0[0]
    old_PO_count = precalculations0[1]
    old_which_POs_appear_wcouple = precalculations0[2]
    old_POs_appear_wcouple = precalculations0[3]
    old_whichcontextPOs_appear_wcouple = precalculations0[4]
    old_position_contextPOs_wcouple = precalculations0[5]
    #We don't assume the old precalculation to have an element with index 6, because it might not have it.

    precalculations1 = [None]*(6+1)#One extra for newcomer_whichcontextPOs_appear_wcouple

    XXXlist=[state[0] for state in scored_states]

    #contextlist=[[op_to_index_dict[op] for op in get_context(XXX)] for XXX in XXXlist]
    newXXX = new_state[0]
    new_context=[op_to_index_dict[op] for op in get_context(newXXX)]
    #Old: #new_context=merge_sort([op_to_index_dict[op] for op in get_context(newXXX)]) #We used to sort the context
    contextlist=old_contextlist+[new_context]#NEWcontextlist
    #Count number of times each PO appears, and whether each couple (both POs) appears.
    PO_count=old_PO_count.copy()#NEWPO_count

    for ii in new_context:
        PO_count[ii]+=1
    newcomer_POs=[]
    for ii in new_context:
        if old_PO_count[ii]==0:
            newcomer_POs.append(ii)

    if 3**nqubit - 1 in newcomer_POs:
        newcomer_POs.remove(3**nqubit - 1)
    newcomer_couplenum=[(int(i/2) if i%2==0 else int((i-1)/2)) for i in newcomer_POs]#Double check here. Continue here
    pre_newcomer_POs_wcouple=[[i*2,i*2+1] for i in newcomer_couplenum if (PO_count[i*2]>0 and PO_count[i*2+1]>0)]

    newcomer_POs_wcouple=np.array(pre_newcomer_POs_wcouple).reshape(2*len(pre_newcomer_POs_wcouple))
    which_POs_appear_wcouple=np.array(list(old_which_POs_appear_wcouple)+list(newcomer_POs_wcouple))#NEWwhich_POs_appear_wcouple
    POs_appear_wcouple=[ii for ii in old_POs_appear_wcouple]
    for i in newcomer_POs_wcouple:
        POs_appear_wcouple[i]=True #NEW_POs_appear_wcouple

    #For each state in the selection: indicate which context POs appear with their couple (does not depend on alpha)
    #whichcontextPOs_appear_wcouple=[None]*len(contextlist)
    contextPOs_appear_wcouple=[None]*len(contextlist)
    for istate,context in enumerate(contextlist):
        does_it_have_couple=np.array([POs_appear_wcouple[i] for i in context],dtype=bool)
        contextPOs_appear_wcouple[istate]=does_it_have_couple
        #whichcontextPOs_appear_wcouple[istate]=np.array(context)[does_it_have_couple]

    #There are some newcomer POs. In the variable "whichcontextPOs_appear_wcouple" we want to know which POs in the
    #context appear with their couple. If the context of state A (context A) has an observable O_1 whose couple did not
    #appear previously, O_1 will not appear in old_whichcontextPOs_appear_wcouple. What could happen for it to be
    #included in new_whichcontextPOs_appear_wcouple? It's couple would need to be in newcomer_POs.

    #Thus, let us first compute what are the couples of the newcomer_POs. We just need them. The first element in a
    #couple is always even. Thus, if we have an even newcomer, add 1. If we have an odd newcomer, subtract 1. That
    #way we obtain their couples.

    newcomer_PO_couples=[((i+1) if i%2==0 else (i-1)) for i in newcomer_POs]

    #intersect_set intersects unsorted lists
    newcomer_whichcontextPOs_appear_wcouple_nolast=[intersect_set(onecontext,newcomer_PO_couples) for onecontext in old_contextlist]

    #Added to prevent index out of bounds (last PO has no couple):
    newstate_context=contextlist[-1]
    if 3**nqubit - 1 in newstate_context:
        newstate_context.remove(3**nqubit - 1)

    newstate_context_couples = [((i+1) if i%2==0 else (i-1)) for i in newstate_context]

    newcomer_whichcontextPOs_appear_wcouple_last=[newstate_context[i] for i,val in enumerate(newstate_context_couples) if PO_count[val]>0]

    #newstate_context_wcouples_couples=[((i+1) if i%2==0 else (i-1)) for i in whichcontextPOs_appear_wcouple_last]

    newcomer_whichcontextPOs_appear_wcouple=newcomer_whichcontextPOs_appear_wcouple_nolast+[newcomer_whichcontextPOs_appear_wcouple_last]

    whichcontextPOs_appear_wcouple=[None]*(len(old_whichcontextPOs_appear_wcouple)+1)#NEWwhichcontextPOs_appear_wcouple
    for i in range(len(old_whichcontextPOs_appear_wcouple)):
        whichcontextPOs_appear_wcouple[i] = np.array([int(po_element) for po_element in np.concatenate((old_whichcontextPOs_appear_wcouple[i],newcomer_whichcontextPOs_appear_wcouple[i]),axis=0)])
        #Above I included a fix for concatenate turning integers into floats when concatenating with empty list.
    whichcontextPOs_appear_wcouple[-1]=newcomer_whichcontextPOs_appear_wcouple_last

    #For each state, this is the position in the context where the POs with appearing couples are.
    position_contextPOs_wcouple=[np.arange(len(contextlist[0]))[appearwcouple] for appearwcouple in contextPOs_appear_wcouple]


    precalculations1[0] = contextlist
    precalculations1[1] = PO_count
    precalculations1[2] = which_POs_appear_wcouple
    precalculations1[3] = POs_appear_wcouple
    precalculations1[4] = whichcontextPOs_appear_wcouple
    precalculations1[5] = position_contextPOs_wcouple
    precalculations1[6] = newcomer_whichcontextPOs_appear_wcouple

    return precalculations1

def precalculate_alpha(scored_states,precalculations,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist,patterns64x64=patterns64x64):
#Code dependent on the alphas

    XXXlist0=[state[0] for state in scored_states]
    precalculations_alpha=[None]*2

    contextlist0 = precalculations[0]
    PO_count = precalculations[1]
    which_POs_appear_wcouple = precalculations[2]
    POs_appear_wcouple = precalculations[3]
    whichcontextPOs_appear_wcouple = precalculations[4]
    position_contextPOs_wcouple = precalculations[5]

    parities_wcouple = [None] * len(contextlist0)

    if len(XXXlist0) < len(contextlist0):
        print("Warning: state selection and precalculation have different size")
        for istate,context in enumerate(contextlist0[:-1]):
            parities_wcouple[istate]=[[PO,parityanalytic(XXXlist0[istate],oplist[PO])] for PO in whichcontextPOs_appear_wcouple[istate]]


        #Calculate the majority parities of the POs that appear with their couples
        global_parities=[[]]*(3**nqubit)
        for state_stats in parities_wcouple[:-1]:
            for state_stat in state_stats:
                global_parities[state_stat[0]]=global_parities[state_stat[0]]+[state_stat[1]]
    else:
        for istate,context in enumerate(contextlist0):
            parities_wcouple[istate]=[[PO,parityanalytic(XXXlist0[istate],oplist[PO])] for PO in whichcontextPOs_appear_wcouple[istate]]

        #Calculate the majority parities of the POs that appear with their couples
        global_parities=[[]]*(3**nqubit)
        for state_stats in parities_wcouple:
            for state_stat in state_stats:
                global_parities[state_stat[0]]=global_parities[state_stat[0]]+[state_stat[1]]

    precalculations_alpha[0] = parities_wcouple
    precalculations_alpha[1] = global_parities

    return precalculations_alpha

def perturb_precalculate_alpha(scored_states,new_state,precalculations,precalculations_alpha0,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist,patterns64x64=patterns64x64):
    #Code dependent on the alphas

    fullscored_states=scored_states+[new_state]
    XXXlist0=[state[0] for state in fullscored_states]
    precalculations_alpha1=[None]*2

    newXXX=new_state[0]

    contextlist0=precalculations[0]
    PO_count=precalculations[1]
    which_POs_appear_wcouple=precalculations[2]
    POs_appear_wcouple=precalculations[3]
    whichcontextPOs_appear_wcouple=precalculations[4]
    position_contextPOs_wcouple=precalculations[5]
    newcomer_whichcontextPOs_appear_wcouple=precalculations[6]

    oldparities_wcouple=[ii for ii in precalculations_alpha0[0]]
    #[[[jj[0],jj[1]] for jj in ii] for ii in precalculations_alpha0[0]]
    oldglobal_parities=[ii for ii in precalculations_alpha0[1]]#This is a shallow copy.
    parities_wcouple=[None]*len(fullscored_states)
    newparities_wcouple=[None]*len(scored_states)
    ################


    for istate,new_contextPOs_wcouple in enumerate(newcomer_whichcontextPOs_appear_wcouple[:-1]):
        newparities_wcouple[istate]=[[PO,parityanalytic(XXXlist0[istate],oplist[PO])] for PO in new_contextPOs_wcouple]
        parities_wcouple[istate]=oldparities_wcouple[istate]+newparities_wcouple[istate]

    parities_wcouple[-1]=[[PO,parityanalytic(XXXlist0[-1],oplist[PO])] for PO in newcomer_whichcontextPOs_appear_wcouple[-1]]

    global_parities=oldglobal_parities
    for state_stats in newparities_wcouple:
        for state_stat in state_stats:
            global_parities[state_stat[0]]=global_parities[state_stat[0]]+[state_stat[1]]

    for state_stat in parities_wcouple[-1]:
        global_parities[state_stat[0]]=global_parities[state_stat[0]]+[state_stat[1]]

    precalculations_alpha1[0] = parities_wcouple
    precalculations_alpha1[1] = global_parities

    return precalculations_alpha1

def perturb_precalculate_multi(scored_states,new_states,precalculations0,precalculations_alpha0,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist,patterns64x64=patterns64x64):
    #This block performs several perturb_precalculate and perturb_precalculate_alpha, starting with the
    #precalculation0 of scored_states, which is a limited selection, and adding one by one the new_states
    #to obtain quickly the precalculation and precalculation_alpha of scored_states+new_states
    precalculation_temp_old=precalculations0
    precalculation_alpha_temp_old=precalculations_alpha0
    scored_states_current=scored_states.copy()

    for ii in range(len(new_states)):
        precalculation_temp = perturb_precalculate(scored_states_current,new_states[ii],precalculation_temp_old)
        precalculation_alpha_temp = perturb_precalculate_alpha(scored_states_current,new_states[ii],precalculation_temp,precalculation_alpha_temp_old)
        precalculation_temp_old = precalculation_temp
        precalculation_alpha_temp_old = precalculation_alpha_temp
        scored_states_current = scored_states_current + [np.array(new_states[ii],dtype=object)]

    return [precalculation_temp,precalculation_alpha_temp]

def alphabeta_improver_nofall_multi5_smartpick(
    scored_states,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64,
    samplealpha=samplealpha,
):
    #We want an improver that does not ever decrease. For that, we'll add and remove one state at a time,
    #but we'll try many many contexts every time.

    #Requirement: the input state selection must be correctly scored!

    #NEXT STEP: I need to incorporate two things into the score:
    #1) newcomer POs in the non-new states,. We need to count the net change in their score.
    #2) Determine which desired_target parities have changed due to the new state. Perhaps this can include 1)
    #    as well.

    #print(extra_states[-3:])

    #Step 1: Perform precalculations for the selection sans the worst-scorer
    scored_states_sorted = list(
        np.array(
            scored_states, dtype=object
        )[np.argsort([-i[1] for i in scored_states])]
    )

    scored_states_sorted_nolast = scored_states_sorted[:-5]

    #print("Starting first precalculations")
    precalculation0 = precalculate(scored_states_sorted_nolast)
    precalculation_alpha0 = precalculate_alpha(
        scored_states_sorted_nolast,
        precalculation0
    )
    #print("Precalculations done")

    basic_PO_count = precalculation0[1]
    basic_missing_POs = [po_id for po_id,val in enumerate(basic_PO_count) if val==0]
    basic_missing_POs_beta = [oplist[po_id] for po_id in basic_missing_POs]

    # Input for each core
    # 1. basic_missing_POs_beta
    # 2.

    #Step 0: Roll 100 random contexts (the alphas don't matter, so we set them to zeros). No duplicates.
    ##Step 0: Roll 100 random contexts (the alphas don't matter, so we set them to zeros). No duplicates.
    numnews = 100 # round(len(scored_states)/8) #I increased the denominator to 100 to speed things up.
    alreadyhere = [i[0][0] for i in scored_states]
    randomindex = []

    pre_fifty_indexes = get_fifty_gens(basic_missing_POs_beta)
    for pre_index in pre_fifty_indexes:
        two_new_indexes = [pre_index[0], pre_index[0]+3**14]
        randomindex = randomindex + two_new_indexes

    randomindex_nodup = [j for j in list(dict.fromkeys(randomindex)) if j not in alreadyhere]
    extra_states_all = [[(ctx_ind,str(int((('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-(nqubit+1):])[0]))+samplealpha,OpString(('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-nqubit:]).complement()),0] for ctx_ind in randomindex_nodup]

    extra_states = extra_states_all + scored_states_sorted[-5:]

    #Step 2: For each context rolled, calculate the best-scorer state. Use the previous precalculations
    #to save resources!
    extra_states2 = [None]*len(extra_states)
    modifiedscores_for_extra_states2 = [None]*len(extra_states)

    #print("Starting second precalculations (using perturb functions)")

    for ii, new_state in enumerate(extra_states):

        precalculation1 = perturb_precalculate(
            scored_states_sorted_nolast,
            new_state,precalculation0
        )
        #precalculation_alpha1=precalculate_alpha(scored_states_sorted_nolast+[np.array(new_state,dtype=object)],precalculation1)#temporary

        #extra_states2[ii]=all_alpha_improver(scored_states_sorted_nolast+[new_state],precalculation1,precalculation_alpha)
        #temp_return=perturb_alpha_improver(scored_states_sorted_nolast+[np.array(new_state,dtype=object)],precalculation1,precalculation_alpha1) #Includes score
        #return temp_return#UNDO this line later


        extra_states2[ii] = perturb_alpha_improver(
            scored_states_sorted_nolast+[np.array(new_state,dtype=object)],
            precalculation1,
            precalculation_alpha1
        ) #Includes score

        precalculate_alpha1_updated = perturb_precalculate_alpha(
            scored_states_sorted_nolast,
            extra_states2[ii][-1],
            precalculation1,
            precalculation_alpha0
        )
        modifiedscores_for_extra_states2[ii]=extra_states2[ii][-1][1]+get_score_modifier(precalculation1,precalculate_alpha1_updated)


    #Step 3: Find the best-best-scorer among contexts.
    #Include it in the selection, kicking out the worst one. Then re-score the selection.

    #bestindex=np.argmax(modifiedscores_for_extra_states2)
    #extra_states2_best=extra_states2[bestindex]

    # Indices of the new states that give you best score. Should havppen after parallel part
    bestindexes = [x[0] for x in sort_bylast([[idx,val] for idx,val in enumerate(modifiedscores_for_extra_states2)])][-5:]
    extra_states2_best = [extra_states2[best_idx] for best_idx in bestindexes]
    # We think these are the scores
    extra_states2_best_justlast = [x[-1] for x in extra_states2_best]
    #return [extra_states2,modifiedscores_for_extra_states2]#We're setting this just to run a test
    precalculation_temp_both = perturb_precalculate_multi(
        scored_states_sorted_nolast,
        extra_states2_best_justlast,
        precalculation0,
        precalculation_alpha0
    )

    best_precalculation = precalculation_temp_both[0]
    best_precalculation_alpha = precalculation_temp_both[1]
    scored_states2 = score_states_quick(
        scored_states_sorted_nolast + extra_states2_best_justlast,
        best_precalculation,
        best_precalculation_alpha
    )


    return scored_states2

def sort_bylast(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using last element of
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[-1])
    return sub_li

def perturb_alpha_improver(scored_states,precalculations,precalculations_alpha,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist,patterns64x64=patterns64x64):

    scored_states2=scored_states.copy()

    single_improved=quick_single_alpha_improver_PROTO(scored_states2[-1],precalculations,precalculations_alpha)

    #OLD:#single_improved=single_alpha_improver(numstates-1,scored_states2[numstates-1],precalculations,precalculations_alpha)
    scored_states2[-1]=np.array([single_improved[0],int(single_improved[1])],dtype=object)

    return scored_states2

#In this block, adapt to make sure that the precalculations are not updated
def quick_single_alpha_improver_PROTO(
    scored_state,
    precalculations,
    precalculations_alpha,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64
):
#This is a PROTOTYPE function. It always improves the last state in the list. scored_state is that state

    #This improvement is meant to modify the alphas of one of the XXXterms. Specifically, the last one.
    ii_state =- 1 #This uses the last state
    XXXterm = scored_state[0]

    #Precalculations
    contextlist0=precalculations[0]
    PO_count=precalculations[1]
    which_POs_appear_wcouple=precalculations[2]
    POs_appear_wcouple=precalculations[3]
    whichcontextPOs_appear_wcouple=precalculations[4]
    position_contextPOs_wcouple=precalculations[5]

    parities_wcouple=[ii for ii in precalculations_alpha[0]]
    global_parities=[ii for ii in precalculations_alpha[1]]#This is a shallow copy.

    parities_wcouple_old=[ii for ii in precalculations_alpha[0]]
    global_parities_old=[ii for ii in precalculations_alpha[1]]#This is a shallow copy.

    desired_target_parities=np.array([None]*len(whichcontextPOs_appear_wcouple[ii_state]))
    for iappear,PO in enumerate(whichcontextPOs_appear_wcouple[ii_state]):
        if PO%2==0:
            complement_mean_parity = round_z(mean_zz(global_parities_old[PO+1]))#REMOVE OLD LATER
            databit = data[int(PO/2)]
        else:
            complement_mean_parity = round_z(mean_zz(global_parities_old[PO-1]))#REMOVE OLD LATER
            databit = data[int((PO-1)/2)]

        if databit==0:
            desired_target_parities[iappear] = complement_mean_parity
        else:
            desired_target_parities[iappear] = 1-complement_mean_parity

    contextsize=2**(nqubit-1)+1
    fullcontext_desired_target_parities=np.array([0.5]*contextsize)

    for ii,ii_context in enumerate(position_contextPOs_wcouple[ii_state]):
        fullcontext_desired_target_parities[ii_context]=desired_target_parities[ii]


    #Compute everything:
    examplecontext=contextlist0[ii_state]
    exampletargetparities=fullcontext_desired_target_parities
    #Adapted from old alpha improver. This uses the fingerprint to calculate the score (though the variable
    #is named Hamming) and pick the best alpha for that score.
    sbit=int(XXXterm[1][0])
    op_gen_to_measure=XXXterm[2]
    op_indexes=examplecontext
    #context_target_parities=[target_parities[op_index] for op_index in op_indexes]
    context_target_parities=fullcontext_desired_target_parities
    t_parities0=exampletargetparities[0]
    #######
    exampletargetparities_pos=position_contextPOs_wcouple[ii_state]
    #######
    exampletargetparities_pos_nozero=(exampletargetparities_pos[1:] if exampletargetparities_pos[0]==0 else exampletargetparities_pos)-1
    #######
    exampletargetparities_pos_nozero_block=[[]]*128
    for i in exampletargetparities_pos_nozero:
        exampletargetparities_pos_nozero_block[int(i//64)]=exampletargetparities_pos_nozero_block[int(i//64)]+[i%64]
    #######
    context_target_parities_block=np.reshape(context_target_parities[1:],[128,64])
    #context_target_parities_block_wcouple=[context_target_parities_block[i_select][select] for i_select,select in enumerate(exampletargetparities_pos_nozero_block)]

    boxed_ponum=[[]]*128
    if position_contextPOs_wcouple[ii_state][0]==0:
        position_contextPOs_wcouple_nozero_shifted=(position_contextPOs_wcouple[ii_state][1:])-1
    else:
        position_contextPOs_wcouple_nozero_shifted=position_contextPOs_wcouple[ii_state]-1
    for cont_ponum in position_contextPOs_wcouple_nozero_shifted:
        boxed_ponum[cont_ponum//64]=boxed_ponum[cont_ponum//64]+[cont_ponum%64]
    context_target_parities_block_wcouple=[np.array([context_target_parities_block[i_box][i] for i in one_boxed_ponum]) for i_box,one_boxed_ponum in enumerate(boxed_ponum)]

    stringlist=["A","B","C","D"]

    stringhamming_precalc=[None]*len(context_target_parities_block_wcouple)
    if ii_state==0:
        temporaloutput=context_target_parities_block_wcouple[0]

    for i, t_par in enumerate(context_target_parities_block_wcouple):

        turntheseon=exampletargetparities_pos_nozero_block[i]
        patterns64x64_restricted=[pattern[:,exampletargetparities_pos_nozero_block[i]] for pattern in patterns64x64]

        stringhamming_precalc[i]=dict([[stringlist[j],np.array([sum(2*(abs(line-t_par)-0.5)) for line in pattern])]
                                 for j,pattern in enumerate(patterns64x64_restricted)])

    firstcol_prehamming=[2*(abs(i-t_parities0)-0.5) for i in first_column]


    #######

    if sbit == 0:
        flattened_prehamming_even=np.reshape([sum([stringhamming_precalc[i][letter] for i,letter in enumerate(blockline)]) for blockline in stringfingerprint_even],(16384))
        flattened_hamming=np.array(flattened_prehamming_even)+np.array(firstcol_prehamming)
    else:
        flattened_prehamming_odd=np.reshape([sum([stringhamming_precalc[i][letter] for i,letter in enumerate(blockline)]) for blockline in stringfingerprint_odd],(16384))
        flattened_hamming=np.array(flattened_prehamming_odd)+np.array(firstcol_prehamming)

    smallest=[0,flattened_hamming[0]]
    for i,value in enumerate(flattened_hamming):
        if value<smallest[1]:
            smallest=[i,value]#[stateindex,hammingvalue]

    newalpha=alphalist[smallest[0]]

    scored_state2=[(XXXterm[0],str(sbit)+newalpha,XXXterm[2]),-smallest[1]]
    #precalculations[0]=0 With this we prove that the precalculations are modified outside as well
    #XXX_new=scored_state2[0]


    return scored_state2

def get_score_modifier(
    temp_precalculation,
    temp_precalculation_alpha,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64
):

    tt_newcomer_whichcontextPOs_appear_wcouple=temp_precalculation[6]
    tt_global_parities=temp_precalculation_alpha[1]

    total_score_modifier = 0
    for PO_index,paritylist in [[ii,tt_global_parities[ii]] for ii in tt_newcomer_whichcontextPOs_appear_wcouple[-1]]:
        avgpar0 = mean_z(paritylist[:-1])
        avgpar1 = mean_z(paritylist)
        shift_direction = paritylist[-1]
        if (avgpar0==0.5 and avgpar1!=0.5) or (avgpar1==0.5 and avgpar0!=0.5):
            maj_shift = True
        elif ((avgpar0>0.5 and avgpar1<0.5) or (avgpar1>0.5 and avgpar0<0.5)):
            maj_shift = True
        else:
            continue

        if PO_index%2==0:
            databit = data[int(PO_index/2)]
        else:
            databit = data[int((PO_index-1)/2)]

        desired_couple_parity= shift_direction
        if databit:
            desired_couple_parity = 1 - desired_couple_parity

        couple_index = ((PO_index+1) if PO_index%2==0 else (PO_index-1))
        couple_paritylist = tt_global_parities[couple_index]
        n_zeros = couple_paritylist.count(0)
        n_ones = couple_paritylist.count(1)

        temp_score_modifier = n_zeros - n_ones
        if desired_couple_parity:
            temp_score_modifier *= -1

        total_score_modifier += temp_score_modifier

    return total_score_modifier

#def score_states_quick(pre_scored_states,precalculations,precalculations_alpha,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist):
#
#    XXXlist0 = [state[0] for state in pre_scored_states]
#
#    contextlist0 = precalculations[0]
#    PO_count = precalculations[1]
#    which_POs_appear_wcouple = precalculations[2]
#    POs_appear_wcouple = precalculations[3]
#    whichcontextPOs_appear_wcouple = precalculations[4]
#    position_contextPOs_wcouple = precalculations[5]
#    ################
#
#
#    #Code dependent on the alphas
#    ################
#    #For each state in the selection: Calculate the parities of the POs which appear with their couple
#    parities_wcouple=precalculations_alpha[0]
#    global_parities=precalculations_alpha[1]
#
#    #Score each state based on the number of POs that appear with their couple with a parity favorable to the data,
#    #using the current majority of the PO couple.
#    state_score=[None]*len(contextlist0)
#    for istate,which_appear in enumerate(whichcontextPOs_appear_wcouple):
#        score=0
#        for iappear,PO in enumerate(which_appear):
#            if PO%2==0:
#                complement_mean_parity=round_z(mean_z(global_parities[PO+1]))
#                databit=data[int(PO/2)]
#            else:
#                complement_mean_parity=round_z(mean_z(global_parities[PO-1]))
#                databit=data[int((PO-1)/2)]
#
#            stateparity=parities_wcouple[istate][iappear][1]
#            if databit==0:
#                if stateparity==complement_mean_parity:
#                    score+=1
#                elif stateparity==(1-complement_mean_parity):
#                    score-=1
#            else:
#                if stateparity==(1-complement_mean_parity):
#                    score+=1
#                elif stateparity==complement_mean_parity:
#                    score-=1
#        state_score[istate]=score
#
#    ################
#
#    scored_states = [[XXXlist0[istate],score] for istate,score in enumerate(state_score)]
#    return scored_states
#
#def score_states_quicker(pre_scored_states,precalculations,precalculations_alpha,data=data,nqubit=nqubit,op_to_index_dict=op_to_index_dict,oplist=oplist):
#
#    #This block is meant to be applied on the pre_scored_states that come out of _PROTO. The last state should have
#    #the correct score already.
#
#    XXXlist0=[state[0] for state in pre_scored_states]
#
#    #Precalculations
#    ################
#    contextlist0=precalculations[0]
#    PO_count=precalculations[1]
#    which_POs_appear_wcouple=precalculations[2]
#    POs_appear_wcouple=precalculations[3]
#    whichcontextPOs_appear_wcouple=precalculations[4]
#    position_contextPOs_wcouple=precalculations[5]
#    newcomer_whichcontextPOs_appear_wcouple=precalculations[6]
#    ################
#
#
#    #Code dependent on the alphas
#    ################
#    #For each state in the selection: Calculate the parities of the POs which appear with their couple
#    parities_wcouple=precalculations_alpha[0]
#    global_parities=precalculations_alpha[1]
#
#    #Score each state based on the number of POs that appear with their couple with a parity favorable to the data,
#    #using the current majority of the PO couple. HOWEVER, we skip the last state, as it was obtained elsewhere.
#    state_score=[None]*len(contextlist0)
#    for istate,which_appear in enumerate(newcomer_whichcontextPOs_appear_wcouple[:-1]):
#        score=0
#        for iappear,PO in enumerate(which_appear):
#            if PO%2==0:
#                complement_mean_parity=round_z(mean_zz(global_parities[PO+1]))
#                databit=data[int(PO/2)]
#            else:
#                complement_mean_parity=round_z(mean_zz(global_parities[PO-1]))
#                databit=data[int((PO-1)/2)]
#
#            stateparity=parityanalytic(XXXlist0[istate],oplist[PO])
#            if databit==0:
#                if stateparity==complement_mean_parity:
#                    score+=1
#                elif stateparity==(1-complement_mean_parity):
#                    score-=1
#            else:
#                if stateparity==(1-complement_mean_parity):
#                    score+=1
#                elif stateparity==complement_mean_parity:
#                    score-=1
#        state_score[istate]=score
#
#    state_score[-1] = 0
#
#    ################
#
#    scored_states=[[XXXlist0[istate],pre_scored_states[istate][1]+score] for istate,score in enumerate(state_score)]
#    return scored_states

def score_gen(missing_pos_beta,gen_beta):
    score=0
    for one_missing in missing_pos_beta:

        compatible=True
        for ii, missing_pauli in enumerate(one_missing):
            if gen_beta[ii]==missing_pauli:
                compatible=False
                break

        if compatible:
            score+=1

    return score

def get_fifty_gens(missing_pos_beta):

    pauli_counter=[[0,0,0] for thing in range(14)]#This is done for 14 qubits specifically
    for one_missing in missing_pos_beta:
        for ii_pauli,one_pauli in enumerate(one_missing):
            if one_pauli=='0':
                pauli_counter[ii_pauli][0]+=1
            elif one_pauli=='1':
                pauli_counter[ii_pauli][1]+=1
            elif one_pauli=='2':
                pauli_counter[ii_pauli][2]+=1

    median_gen="".join([str(np.argmin(thing)) for thing in pauli_counter])
    median_gen_list=[thing for thing in median_gen]

    top_twenty_sorted = [[['0','0','0'], 0] for thing in range(50)]
    how_many_to_change = 7

    for ii_counter in range(1000):

        median_gen_list_copy = [thing for thing in median_gen_list]

        for ii in range(how_many_to_change):
            median_gen_list_copy[random.randint(0,12)]=str(random.randint(0,2))
        top_twenty_sorted_only_gens=["".join(thing[0]) for thing in top_twenty_sorted]
        this_gen="".join(median_gen_list_copy)
        if this_gen in top_twenty_sorted_only_gens:
            temp_score=0
        else:
            temp_score=score_gen(missing_pos_beta,median_gen_list_copy)
        if temp_score>top_twenty_sorted[0][1]:
            top_twenty_sorted[0]=[[thing for thing in median_gen_list_copy],temp_score]
            top_twenty_sorted=sort_bylast(top_twenty_sorted)

    top_generators=[[op_to_index_dict["".join(i[0])],i[1]] for i in top_twenty_sorted]

    return top_generators

def parallel_part(
    basic_missing_POs_beta,
    nqubit: int,
    numnews: int=100
):
    alreadyhere = [i[0][0] for i in scored_states]
    randomindex = []

    pre_fifty_indexes = get_fifty_gens(basic_missing_POs_beta)
    for pre_index in pre_fifty_indexes:
        two_new_indexes = [pre_index[0], pre_index[0]+3**nqubit]
        randomindex = randomindex + two_new_indexes

    randomindex_nodup = [j for j in list(dict.fromkeys(randomindex)) if j not in alreadyhere]
    extra_states_all = [[(ctx_ind,str(int((('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-(nqubit+1):])[0]))+samplealpha,OpString(('0'*(nqubit)+convert_basis(ctx_ind,10,3))[-nqubit:]).complement()),0] for ctx_ind in randomindex_nodup]

    extra_states = extra_states_all + scored_states_sorted[-5:]

    #Step 2: For each context rolled, calculate the best-scorer state. Use the previous precalculations
    #to save resources!
    extra_states2=[None]*len(extra_states)
    modifiedscores_for_extra_states2=[None]*len(extra_states)
    for ii, new_state in enumerate(extra_states):
        precalculation1 = perturb_precalculate(
            scored_states_sorted_nolast,
            new_state,precalculation0
        )
        extra_states2[ii] = perturb_alpha_improver(
            scored_states_sorted_nolast+[np.array(new_state,dtype=object)],
            precalculation1,
            precalculation_alpha1
        )

        precalculate_alpha1_updated = perturb_precalculate_alpha(
            scored_states_sorted_nolast,
            extra_states2[ii][-1],
            precalculation1,
            precalculation_alpha0
        )
        modifiedscores_for_extra_states2[ii]=extra_states2[ii][-1][1]+get_score_modifier(precalculation1,precalculate_alpha1_updated)


    #Step 3: Find the best-best-scorer among contexts.
    #Include it in the selection, kicking out the worst one. Then re-score the selection.
    bestindexes = [thing[0] for thing in sort_bylast([[idx,val] for idx,val in enumerate(modifiedscores_for_extra_states2)])][-5:]
    extra_states2_best=[extra_states2[best_idx] for best_idx in bestindexes]
    extra_states2_best_justlast=[one[-1] for one in extra_states2_best]

    precalculation_temp_both = perturb_precalculate_multi(
        scored_states_sorted_nolast,
        extra_states2_best_justlast,
        precalculation0,
        precalculation_alpha0
    )

    best_precalculation = precalculation_temp_both[0]
    best_precalculation_alpha = precalculation_temp_both[1]
    scored_states2 = score_states_quick(scored_states_sorted_nolast+extra_states2_best_justlast,best_precalculation,best_precalculation_alpha)
    return scored_states2

def cool_test_function(
    scored_states,
    data=data,
    nqubit=nqubit,
    op_to_index_dict=op_to_index_dict,
    oplist=oplist,
    patterns64x64=patterns64x64,
    samplealpha=samplealpha,
    num_cores=1
):
    #Step 1: Perform precalculations for the selection sans the worst-scorer
    scored_states_sorted = list(
        np.array(
            scored_states, dtype=object
        )[np.argsort([-i[1] for i in scored_states])]
    )

    scored_states_sorted_nolast = scored_states_sorted[:-5]

    precalculation0 = precalculate(scored_states_sorted_nolast)
    precalculation_alpha0 = precalculate_alpha(
        scored_states_sorted_nolast,
        precalculation0
    )

    basic_PO_count = precalculation0[1]
    basic_missing_POs = [po_id for po_id,val in enumerate(basic_PO_count) if val==0]
    basic_missing_POs_beta = [oplist[po_id] for po_id in basic_missing_POs]

    output = np.full(scored_states2.shape + (num_cores,), np.nan)

    for idx in range(num_cores):
        x = parallel_part(basic_missing_POs_beta, nqubit)
        output[:, :, idx] = x

    return output

start = time()

previous_stateselection_list=[
    i for i in np.load(f"{save_folder}/scored_states_nofall_multi_smartpick_ith.npy", allow_pickle=True)
]

scored_states_nofall_multi_smartpick_ith = [None] * (4+1)
scored_states_nofall_multi_smartpick_ith[0] = previous_stateselection_list[-1]

currentscore=sum([i[1] for i in scored_states_nofall_multi_smartpick_ith[0]])

print("Selection", (len(previous_stateselection_list)-1),"score: ", currentscore)
for ii in tqdm(range(4)):
    #scored_states_nofall_multi_smartpick_ith[ii+1] = alphabeta_improver_nofall_multi_smartpick(scored_states_nofall_multi_smartpick_ith[ii])
    scored_states_nofall_multi_smartpick_ith[ii+1] = (
        alphabeta_improver_nofall_multi5_smartpick(scored_states_nofall_multi_smartpick_ith[ii])
    )

    # output = new_function(scored_states_nofall_multi_smartpick_ith[ii], num_cores=NUM_CORES)
    # Sort output by score
    # sorted(output, lambda x: -score(x))
    # Make sure that no duplicates on idx.
    # best_five = []
    # while len(best_file) < 5:
        # check if idx already in there
        # if not add it

    # Swap worst states for best five if scores are in fact better

    # Rescore the states using the quick score method


    currentscore = sum([state[1] for state in scored_states_nofall_multi_smartpick_ith[ii+1]])

    print("Selection", (len(previous_stateselection_list)+ii),"score: ",currentscore)

    np.save(
        save_folder+"scored_states_nofall_multi_smartpick_ith.npy",
        np.array((previous_stateselection_list+scored_states_nofall_multi_smartpick_ith[1:ii+2]),dtype=object)
    )
    np.save(
        save_folder+"scored_states_nofall_multi_smartpick_ith_backup.npy",
        np.array((previous_stateselection_list+scored_states_nofall_multi_smartpick_ith[1:ii+2]),dtype=object)
    )
end = time()
print("")
print("time:", end-start)
