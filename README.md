CURIS 2021 Project


Question:

Minimize the expected size of an SIR outbreak (in either the Markovian or percolation model) given an arbitrary graph G and a budget of vaccine/tests/restrictions which reduce transmission rates on edges or increase recovery rates at nodes.

Background:

Several works have shown algorithms for modifying transmission/recovery rates which serve to minimize the size of an outbreak or prevent a linear-sized SIR or exponentially long SIS epidemic. Most works only show that a particular algorithm prevents an epidemic [3], or upper bounds the number of new infected nodes given that the outbreak is exponentially decreasing. [1, see also 2 for SIS]

We seek to analyze and improve upon these algorithms in two ways:
Understand the optimality of SOTA algorithms in preventing an epidemic. Can we show that in certain graphs or all graphs the budget given by the convex programming based algorithms in [1] or [2] for minimizing small outbreaks is near-optimal? If not, can we design new algorithms which can prevent an epidemic with a smaller budget in some or all graphs? Alternatively, can we show that the degree-proportional allocation strategy in [3] is near-optimal or far from optimal in non-expander graphs.
Understand what happens above the epidemic threshold. Are the SOTA algorithms applicable and useful above the epidemic threshold (that is, when the outbreak is not exponentially decreasing)? What other algorithms can yield better guarantees for some or all graphs? (eg, graphs with strong community structure)

[1] Efficient Containment of Exact SIR Markovian Processes on Networks 

[2] Optimal Resource Allocation for Network Protection Against Spreading Processes 

[3] How to distribute antidote to control epidemics - Borgs - 2010 - Random Structures & Algorithms 
