 
åČ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
„
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8Åü

 Adam/user_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\
*1
shared_name" Adam/user_embedding/embeddings/v

4Adam/user_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/v*
_output_shapes
:	\
*
dtype0

 Adam/book_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N
*1
shared_name" Adam/book_embedding/embeddings/v

4Adam/book_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/book_embedding/embeddings/v*
_output_shapes
:	N
*
dtype0

 Adam/user_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\
*1
shared_name" Adam/user_embedding/embeddings/m

4Adam/user_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/m*
_output_shapes
:	\
*
dtype0

 Adam/book_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N
*1
shared_name" Adam/book_embedding/embeddings/m

4Adam/book_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/book_embedding/embeddings/m*
_output_shapes
:	N
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

user_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\
**
shared_nameuser_embedding/embeddings

-user_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_embedding/embeddings*
_output_shapes
:	\
*
dtype0

book_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N
**
shared_namebook_embedding/embeddings

-book_embedding/embeddings/Read/ReadVariableOpReadVariableOpbook_embedding/embeddings*
_output_shapes
:	N
*
dtype0
}
serving_default_book_inputPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
}
serving_default_user_inputPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_book_inputserving_default_user_inputuser_embedding/embeddingsbook_embedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3515365

NoOpNoOp
ų(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³(
value©(B¦( B(
Ū
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

_init_input_shape* 
* 
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 

,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 
°
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
7trace_0
8trace_1
9trace_2
:trace_3* 
6
;trace_0
<trace_1
=trace_2
>trace_3* 
* 
h
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemmmnvovp*

Dserving_default* 
* 

0*

0*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
mg
VARIABLE_VALUEbook_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 
mg
VARIABLE_VALUEuser_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

Xtrace_0* 

Ytrace_0* 
* 
* 
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

_trace_0* 

`trace_0* 
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

ftrace_0* 

gtrace_0* 
* 
5
0
1
2
3
4
5
6*

h0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
i	variables
j	keras_api
	ktotal
	lcount*

k0
l1*

i	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/book_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/user_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/book_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/user_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¾
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-book_embedding/embeddings/Read/ReadVariableOp-user_embedding/embeddings/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/book_embedding/embeddings/m/Read/ReadVariableOp4Adam/user_embedding/embeddings/m/Read/ReadVariableOp4Adam/book_embedding/embeddings/v/Read/ReadVariableOp4Adam/user_embedding/embeddings/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3515578
µ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebook_embedding/embeddingsuser_embedding/embeddings	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/book_embedding/embeddings/m Adam/user_embedding/embeddings/m Adam/book_embedding/embeddings/v Adam/user_embedding/embeddings/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3515627¤©
Ķ
¬
)__inference_model_5_layer_call_fn_3515375
inputs_0
inputs_1
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_3515233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1
Ł
°
)__inference_model_5_layer_call_fn_3515319

user_input

book_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCall
user_input
book_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_3515302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input
æ
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515497

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
¬	
Ŗ
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198

inputs+
embedding_lookup_3515192:	N

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’½
embedding_lookupResourceGatherembedding_lookup_3515192Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3515192*+
_output_shapes
:’’’’’’’’’
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3515192*+
_output_shapes
:’’’’’’’’’

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Å

D__inference_model_5_layer_call_and_return_conditional_losses_3515333

user_input

book_input)
user_embedding_3515323:	\
)
book_embedding_3515326:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallł
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_3515323*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184ł
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_3515326*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198ę
flatten_11/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208ę
flatten_10/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216ö
dot_5/PartitionedCallPartitionedCall#flatten_11/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230m
IdentityIdentitydot_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input
Ł
°
)__inference_model_5_layer_call_fn_3515240

user_input

book_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCall
user_input
book_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_3515233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input
±

D__inference_model_5_layer_call_and_return_conditional_losses_3515302

inputs
inputs_1)
user_embedding_3515292:	\
)
book_embedding_3515295:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallõ
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_3515292*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184÷
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_3515295*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198ę
flatten_11/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208ę
flatten_10/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216ö
dot_5/PartitionedCallPartitionedCall#flatten_11/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230m
IdentityIdentitydot_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¬	
Ŗ
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515475

inputs+
embedding_lookup_3515469:	\

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’½
embedding_lookupResourceGatherembedding_lookup_3515469Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3515469*+
_output_shapes
:’’’’’’’’’
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3515469*+
_output_shapes
:’’’’’’’’’

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¬	
Ŗ
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184

inputs+
embedding_lookup_3515178:	\

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’½
embedding_lookupResourceGatherembedding_lookup_3515178Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3515178*+
_output_shapes
:’’’’’’’’’
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3515178*+
_output_shapes
:’’’’’’’’’

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

S
'__inference_dot_5_layer_call_fn_3515503
inputs_0
inputs_1
identityŗ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’
:’’’’’’’’’
:Q M
'
_output_shapes
:’’’’’’’’’

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’

"
_user_specified_name
inputs/1
Ī
Æ
D__inference_model_5_layer_call_and_return_conditional_losses_3515413
inputs_0
inputs_1:
'user_embedding_embedding_lookup_3515390:	\
:
'book_embedding_embedding_lookup_3515396:	N

identity¢book_embedding/embedding_lookup¢user_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ł
user_embedding/embedding_lookupResourceGather'user_embedding_embedding_lookup_3515390user_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@user_embedding/embedding_lookup/3515390*+
_output_shapes
:’’’’’’’’’
*
dtype0Š
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@user_embedding/embedding_lookup/3515390*+
_output_shapes
:’’’’’’’’’

*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
f
book_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ł
book_embedding/embedding_lookupResourceGather'book_embedding_embedding_lookup_3515396book_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@book_embedding/embedding_lookup/3515396*+
_output_shapes
:’’’’’’’’’
*
dtype0Š
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@book_embedding/embedding_lookup/3515396*+
_output_shapes
:’’’’’’’’’

*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_11/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_10/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
V
dot_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_5/ExpandDims
ExpandDimsflatten_11/Reshape:output:0dot_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
X
dot_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_5/ExpandDims_1
ExpandDimsflatten_10/Reshape:output:0dot_5/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’

dot_5/MatMulBatchMatMulV2dot_5/ExpandDims:output:0dot_5/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’P
dot_5/ShapeShapedot_5/MatMul:output:0*
T0*
_output_shapes
:x
dot_5/SqueezeSqueezedot_5/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
e
IdentityIdentitydot_5/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp ^book_embedding/embedding_lookup ^user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2B
book_embedding/embedding_lookupbook_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1
«	
l
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’
:’’’’’’’’’
:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
é 
±
"__inference__wrapped_model_3515165

user_input

book_inputB
/model_5_user_embedding_embedding_lookup_3515142:	\
B
/model_5_book_embedding_embedding_lookup_3515148:	N

identity¢'model_5/book_embedding/embedding_lookup¢'model_5/user_embedding/embedding_lookupp
model_5/user_embedding/CastCast
user_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’
'model_5/user_embedding/embedding_lookupResourceGather/model_5_user_embedding_embedding_lookup_3515142model_5/user_embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@model_5/user_embedding/embedding_lookup/3515142*+
_output_shapes
:’’’’’’’’’
*
dtype0č
0model_5/user_embedding/embedding_lookup/IdentityIdentity0model_5/user_embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@model_5/user_embedding/embedding_lookup/3515142*+
_output_shapes
:’’’’’’’’’
Æ
2model_5/user_embedding/embedding_lookup/Identity_1Identity9model_5/user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
p
model_5/book_embedding/CastCast
book_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’
'model_5/book_embedding/embedding_lookupResourceGather/model_5_book_embedding_embedding_lookup_3515148model_5/book_embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@model_5/book_embedding/embedding_lookup/3515148*+
_output_shapes
:’’’’’’’’’
*
dtype0č
0model_5/book_embedding/embedding_lookup/IdentityIdentity0model_5/book_embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@model_5/book_embedding/embedding_lookup/3515148*+
_output_shapes
:’’’’’’’’’
Æ
2model_5/book_embedding/embedding_lookup/Identity_1Identity9model_5/book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
i
model_5/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   ·
model_5/flatten_11/ReshapeReshape;model_5/book_embedding/embedding_lookup/Identity_1:output:0!model_5/flatten_11/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
i
model_5/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   ·
model_5/flatten_10/ReshapeReshape;model_5/user_embedding/embedding_lookup/Identity_1:output:0!model_5/flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
^
model_5/dot_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
model_5/dot_5/ExpandDims
ExpandDims#model_5/flatten_11/Reshape:output:0%model_5/dot_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
`
model_5/dot_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :¬
model_5/dot_5/ExpandDims_1
ExpandDims#model_5/flatten_10/Reshape:output:0'model_5/dot_5/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
£
model_5/dot_5/MatMulBatchMatMulV2!model_5/dot_5/ExpandDims:output:0#model_5/dot_5/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’`
model_5/dot_5/ShapeShapemodel_5/dot_5/MatMul:output:0*
T0*
_output_shapes
:
model_5/dot_5/SqueezeSqueezemodel_5/dot_5/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
m
IdentityIdentitymodel_5/dot_5/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp(^model_5/book_embedding/embedding_lookup(^model_5/user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2R
'model_5/book_embedding/embedding_lookup'model_5/book_embedding/embedding_lookup2R
'model_5/user_embedding/embedding_lookup'model_5/user_embedding/embedding_lookup:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input
±

D__inference_model_5_layer_call_and_return_conditional_losses_3515233

inputs
inputs_1)
user_embedding_3515185:	\
)
book_embedding_3515199:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallõ
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_3515185*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184÷
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_3515199*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198ę
flatten_11/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208ę
flatten_10/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216ö
dot_5/PartitionedCallPartitionedCall#flatten_11/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230m
IdentityIdentitydot_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
&

 __inference__traced_save_3515578
file_prefix8
4savev2_book_embedding_embeddings_read_readvariableop8
4savev2_user_embedding_embeddings_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_book_embedding_embeddings_m_read_readvariableop?
;savev2_adam_user_embedding_embeddings_m_read_readvariableop?
;savev2_adam_book_embedding_embeddings_v_read_readvariableop?
;savev2_adam_user_embedding_embeddings_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ³
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ü
valueŅBĻB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_book_embedding_embeddings_read_readvariableop4savev2_user_embedding_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_book_embedding_embeddings_m_read_readvariableop;savev2_adam_user_embedding_embeddings_m_read_readvariableop;savev2_adam_book_embedding_embeddings_v_read_readvariableop;savev2_adam_user_embedding_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*g
_input_shapesV
T: :	N
:	\
: : : : : : : :	N
:	\
:	N
:	\
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	N
:%!

_output_shapes
:	\
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	N
:%!

_output_shapes
:	\
:%!

_output_shapes
:	N
:%!

_output_shapes
:	\
:

_output_shapes
: 
±

0__inference_book_embedding_layer_call_fn_3515448

inputs
unknown:	N

identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
¬
)__inference_model_5_layer_call_fn_3515385
inputs_0
inputs_1
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_3515302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1
¬	
Ŗ
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515458

inputs+
embedding_lookup_3515452:	N

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’½
embedding_lookupResourceGatherembedding_lookup_3515452Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3515452*+
_output_shapes
:’’’’’’’’’
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3515452*+
_output_shapes
:’’’’’’’’’

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³	
n
B__inference_dot_5_layer_call_and_return_conditional_losses_3515515
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :q

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’
:’’’’’’’’’
:Q M
'
_output_shapes
:’’’’’’’’’

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’

"
_user_specified_name
inputs/1
æ
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515486

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
æ
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
«
H
,__inference_flatten_11_layer_call_fn_3515480

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
Ī
Æ
D__inference_model_5_layer_call_and_return_conditional_losses_3515441
inputs_0
inputs_1:
'user_embedding_embedding_lookup_3515418:	\
:
'book_embedding_embedding_lookup_3515424:	N

identity¢book_embedding/embedding_lookup¢user_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ł
user_embedding/embedding_lookupResourceGather'user_embedding_embedding_lookup_3515418user_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@user_embedding/embedding_lookup/3515418*+
_output_shapes
:’’’’’’’’’
*
dtype0Š
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@user_embedding/embedding_lookup/3515418*+
_output_shapes
:’’’’’’’’’

*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
f
book_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ł
book_embedding/embedding_lookupResourceGather'book_embedding_embedding_lookup_3515424book_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@book_embedding/embedding_lookup/3515424*+
_output_shapes
:’’’’’’’’’
*
dtype0Š
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@book_embedding/embedding_lookup/3515424*+
_output_shapes
:’’’’’’’’’

*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_11/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_10/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
V
dot_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_5/ExpandDims
ExpandDimsflatten_11/Reshape:output:0dot_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
X
dot_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_5/ExpandDims_1
ExpandDimsflatten_10/Reshape:output:0dot_5/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’

dot_5/MatMulBatchMatMulV2dot_5/ExpandDims:output:0dot_5/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’P
dot_5/ShapeShapedot_5/MatMul:output:0*
T0*
_output_shapes
:x
dot_5/SqueezeSqueezedot_5/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
e
IdentityIdentitydot_5/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp ^book_embedding/embedding_lookup ^user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2B
book_embedding/embedding_lookupbook_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1
±

0__inference_user_embedding_layer_call_fn_3515465

inputs
unknown:	\

identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
³
¬
%__inference_signature_wrapper_3515365

book_input

user_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCall
user_input
book_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_3515165o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input
Å

D__inference_model_5_layer_call_and_return_conditional_losses_3515347

user_input

book_input)
user_embedding_3515337:	\
)
book_embedding_3515340:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallł
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_3515337*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515184ł
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_3515340*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515198ę
flatten_11/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515208ę
flatten_10/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216ö
dot_5/PartitionedCallPartitionedCall#flatten_11/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dot_5_layer_call_and_return_conditional_losses_3515230m
IdentityIdentitydot_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
user_input:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
book_input
ł7

#__inference__traced_restore_3515627
file_prefix=
*assignvariableop_book_embedding_embeddings:	N
?
,assignvariableop_1_user_embedding_embeddings:	\
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: F
3assignvariableop_9_adam_book_embedding_embeddings_m:	N
G
4assignvariableop_10_adam_user_embedding_embeddings_m:	\
G
4assignvariableop_11_adam_book_embedding_embeddings_v:	N
G
4assignvariableop_12_adam_user_embedding_embeddings_v:	\

identity_14¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¶
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ü
valueŅBĻB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_book_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp,assignvariableop_1_user_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_9AssignVariableOp3assignvariableop_9_adam_book_embedding_embeddings_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_10AssignVariableOp4assignvariableop_10_adam_user_embedding_embeddings_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_book_embedding_embeddings_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_user_embedding_embeddings_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ķ
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ś
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
«
H
,__inference_flatten_10_layer_call_fn_3515491

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515216`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’
:S O
+
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ń
serving_defaultŻ
A

book_input3
serving_default_book_input:0’’’’’’’’’
A

user_input3
serving_default_user_input:0’’’’’’’’’9
dot_50
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Č
ņ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
"
_tf_keras_input_layer
µ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
µ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
„
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
„
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
„
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ł
7trace_0
8trace_1
9trace_2
:trace_32ī
)__inference_model_5_layer_call_fn_3515240
)__inference_model_5_layer_call_fn_3515375
)__inference_model_5_layer_call_fn_3515385
)__inference_model_5_layer_call_fn_3515319æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z7trace_0z8trace_1z9trace_2z:trace_3
Å
;trace_0
<trace_1
=trace_2
>trace_32Ś
D__inference_model_5_layer_call_and_return_conditional_losses_3515413
D__inference_model_5_layer_call_and_return_conditional_losses_3515441
D__inference_model_5_layer_call_and_return_conditional_losses_3515333
D__inference_model_5_layer_call_and_return_conditional_losses_3515347æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z;trace_0z<trace_1z=trace_2z>trace_3
ÜBŁ
"__inference__wrapped_model_3515165
user_input
book_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
w
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemmmnvovp"
	optimizer
,
Dserving_default"
signature_map
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ō
Jtrace_02×
0__inference_book_embedding_layer_call_fn_3515448¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zJtrace_0

Ktrace_02ņ
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515458¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zKtrace_0
,:*	N
2book_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ō
Qtrace_02×
0__inference_user_embedding_layer_call_fn_3515465¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zQtrace_0

Rtrace_02ņ
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515475¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zRtrace_0
,:*	\
2user_embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
š
Xtrace_02Ó
,__inference_flatten_11_layer_call_fn_3515480¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zXtrace_0

Ytrace_02ī
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515486¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zYtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
š
_trace_02Ó
,__inference_flatten_10_layer_call_fn_3515491¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z_trace_0

`trace_02ī
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515497¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z`trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ė
ftrace_02Ī
'__inference_dot_5_layer_call_fn_3515503¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zftrace_0

gtrace_02é
B__inference_dot_5_layer_call_and_return_conditional_losses_3515515¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zgtrace_0
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_model_5_layer_call_fn_3515240
user_input
book_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
)__inference_model_5_layer_call_fn_3515375inputs/0inputs/1"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
)__inference_model_5_layer_call_fn_3515385inputs/0inputs/1"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
)__inference_model_5_layer_call_fn_3515319
user_input
book_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
”B
D__inference_model_5_layer_call_and_return_conditional_losses_3515413inputs/0inputs/1"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
”B
D__inference_model_5_layer_call_and_return_conditional_losses_3515441inputs/0inputs/1"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
„B¢
D__inference_model_5_layer_call_and_return_conditional_losses_3515333
user_input
book_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
„B¢
D__inference_model_5_layer_call_and_return_conditional_losses_3515347
user_input
book_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ŁBÖ
%__inference_signature_wrapper_3515365
book_input
user_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBį
0__inference_book_embedding_layer_call_fn_3515448inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
’Bü
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515458inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBį
0__inference_user_embedding_layer_call_fn_3515465inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
’Bü
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515475inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ąBŻ
,__inference_flatten_11_layer_call_fn_3515480inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ūBų
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515486inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ąBŻ
,__inference_flatten_10_layer_call_fn_3515491inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ūBų
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515497inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ēBä
'__inference_dot_5_layer_call_fn_3515503inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B’
B__inference_dot_5_layer_call_and_return_conditional_losses_3515515inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
N
i	variables
j	keras_api
	ktotal
	lcount"
_tf_keras_metric
.
k0
l1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
1:/	N
2 Adam/book_embedding/embeddings/m
1:/	\
2 Adam/user_embedding/embeddings/m
1:/	N
2 Adam/book_embedding/embeddings/v
1:/	\
2 Adam/user_embedding/embeddings/vŗ
"__inference__wrapped_model_3515165^¢[
T¢Q
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
Ŗ "-Ŗ*
(
dot_5
dot_5’’’’’’’’’®
K__inference_book_embedding_layer_call_and_return_conditional_losses_3515458_/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’

 
0__inference_book_embedding_layer_call_fn_3515448R/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
Ź
B__inference_dot_5_layer_call_and_return_conditional_losses_3515515Z¢W
P¢M
KH
"
inputs/0’’’’’’’’’

"
inputs/1’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’
 ”
'__inference_dot_5_layer_call_fn_3515503vZ¢W
P¢M
KH
"
inputs/0’’’’’’’’’

"
inputs/1’’’’’’’’’

Ŗ "’’’’’’’’’§
G__inference_flatten_10_layer_call_and_return_conditional_losses_3515497\3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 
,__inference_flatten_10_layer_call_fn_3515491O3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
§
G__inference_flatten_11_layer_call_and_return_conditional_losses_3515486\3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 
,__inference_flatten_11_layer_call_fn_3515480O3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
Ü
D__inference_model_5_layer_call_and_return_conditional_losses_3515333f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ü
D__inference_model_5_layer_call_and_return_conditional_losses_3515347f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ų
D__inference_model_5_layer_call_and_return_conditional_losses_3515413b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ų
D__inference_model_5_layer_call_and_return_conditional_losses_3515441b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 “
)__inference_model_5_layer_call_fn_3515240f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’“
)__inference_model_5_layer_call_fn_3515319f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’°
)__inference_model_5_layer_call_fn_3515375b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’°
)__inference_model_5_layer_call_fn_3515385b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p

 
Ŗ "’’’’’’’’’Ō
%__inference_signature_wrapper_3515365Ŗu¢r
¢ 
kŖh
2

book_input$!

book_input’’’’’’’’’
2

user_input$!

user_input’’’’’’’’’"-Ŗ*
(
dot_5
dot_5’’’’’’’’’®
K__inference_user_embedding_layer_call_and_return_conditional_losses_3515475_/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’

 
0__inference_user_embedding_layer_call_fn_3515465R/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
