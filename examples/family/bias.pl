max_clauses(4).
max_vars(3).
max_body(2).
enable_pi.

head_pred(daughter,2).
body_pred(parent,2).
body_pred(female,1).

type(daughter,(person,person)).
type(parent,(person,person)).
type(female,(person,)).
