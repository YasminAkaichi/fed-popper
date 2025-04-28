max_clauses(3).
max_vars(5).
max_body(6).

head_pred(active,1).
body_pred(lumo,2).
body_pred(logp,2).
body_pred(bond,4).
body_pred(atm,5).
body_pred(gteq,2).
body_pred(lteq,2).
%body_pred(eq,2).
body_pred(benzene,2).
body_pred(carbon_5_aromatic_ring,2).
body_pred(carbon_6_ring,2).
body_pred(hetero_aromatic_6_ring,2).
body_pred(hetero_aromatic_5_ring,2).
body_pred(ring_size_6,2).
body_pred(ring_size_5,2).
body_pred(nitro,2).
body_pred(methyl,2).
body_pred(anthracene,2).
body_pred(phenanthrene,2).
body_pred(ball3,2).
body_pred(member,2).
%body_pred(connected,2).

type(active,(drug,)).
type(lumo,(drug,float)).
type(logp,(drug,float)).
type(bond,(drug,atomid,atomid,float)).
type(atm,(drug,atomid,element,float,charge)).
type(gteq,(float,float)).
type(lteq,(float,float)).
%type(eq,(charge,charge,)).
%type(eq,(energy,energy,)).
%type(eq,(hydrophob,hydrophob,)).
type(benzene,(drug,ring)).
type(carbon_5_aromatic_ring,(drug,ring)).
type(carbon_6_ring,(drug,ring)).
type(hetero_aromatic_6_ring,(drug,ring)).
type(hetero_aromatic_5_ring,(drug,ring)).
type(ring_size_6,(drug,ring)).
type(ring_size_5,(drug,ring)).
type(nitro,(drug,ring)).
type(methyl,(drug,ring)).
type(anthracene,(drug,ringlist)).
type(phenanthrene,(drug,ringlist)).
type(ball3,(drug,ringlist)).
type(member,(ring,ringlist)).
type(connected,(ring,ring)).

direction(active,(in,)).
direction(lumo,(in,out)).
direction(logp,(in,out)).
direction(bond,(in,out,out,out)).
direction(bond,(in,in,out,out)).
direction(atm,(in,out,out,out,out)).
direction(gteq,(in,out)).
direction(lteq,(in,out)).
direction(eq,(in,out)).
direction(benzene,(in,out)).
direction(carbon_5_aromatic_ring,(in,out)).
direction(carbon_6_ring,(in,out)).
direction(hetero_aromatic_6_ring,(in,out)).
direction(hetero_aromatic_5_ring,(in,out)).
direction(ring_size_6,(in,out)).
direction(ring_size_5,(in,out)).
direction(nitro,(in,out)).
direction(methyl,(in,out)).
direction(anthracene,(in,out)).
direction(phenanthrene,(in,out)).
direction(ball3,(in,out)).
direction(ball3,(in,in)).
direction(connected,(in,in)).
direction(member,(out,in)).


constant(br,(element,)).
constant(c,(element,)).
constant(cl,(element,)).
constant(f,(element,)).
constant(h,(element,)).
constant(i,(element,)).
constant(n,(element,)).
constant(o,(element,)).
constant(s,(element,)).

body_pred(P,1):-
	constant(P,_).

type(P,(T,)):-
	constant(P,T).

direction(P,(out,)):-
	constant(P,_).


body_pred(ind1,2).
type(ind1,(drug,float)).
direction(ind1,(out,in)).

body_pred(inda,2).
type(inda,(drug,float)).
direction(inda,(out,in)).

body_pred(act,2).
type(act,(drug,float)).
direction(act,(out,in)).

body_pred(zero,1).
type(zero,(float,)).
direction(zero,(out,)).


:-
	clause(C),
	#count{V : clause_var(C,V),var_type(C,V,drug)} != 1.