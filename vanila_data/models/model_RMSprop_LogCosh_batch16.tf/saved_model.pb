ņ
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
 "serve*2.10.02unknown8ė
§
%RMSprop/user_embedding/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\
*6
shared_name'%RMSprop/user_embedding/embeddings/rms
 
9RMSprop/user_embedding/embeddings/rms/Read/ReadVariableOpReadVariableOp%RMSprop/user_embedding/embeddings/rms*
_output_shapes
:	\
*
dtype0
§
%RMSprop/book_embedding/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N
*6
shared_name'%RMSprop/book_embedding/embeddings/rms
 
9RMSprop/book_embedding/embeddings/rms/Read/ReadVariableOpReadVariableOp%RMSprop/book_embedding/embeddings/rms*
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
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
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

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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_14105397

NoOpNoOp
Ü&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*&
value&B& B&
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
Y
?iter
	@decay
Alearning_rate
Bmomentum
Crho	rmsm	rmsn*
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
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
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

VARIABLE_VALUE%RMSprop/book_embedding/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%RMSprop/user_embedding/embeddings/rmsXlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
é
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-book_embedding/embeddings/Read/ReadVariableOp-user_embedding/embeddings/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9RMSprop/book_embedding/embeddings/rms/Read/ReadVariableOp9RMSprop/user_embedding/embeddings/rms/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_14105604

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebook_embedding/embeddingsuser_embedding/embeddingsRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcount%RMSprop/book_embedding/embeddings/rms%RMSprop/user_embedding/embeddings/rms*
Tin
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_14105647£
Ó
£
F__inference_model_17_layer_call_and_return_conditional_losses_14105365

user_input

book_input*
user_embedding_14105355:	\
*
book_embedding_14105358:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallū
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_14105355*
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
GPU 2J 8 *U
fPRN
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216ū
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_14105358*
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
GPU 2J 8 *U
fPRN
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230ē
flatten_35/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240ē
flatten_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248ł
dot_17/PartitionedCallPartitionedCall#flatten_35/PartitionedCall:output:0#flatten_34/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262n
IdentityIdentitydot_17/PartitionedCall:output:0^NoOp*
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
µ
­
&__inference_signature_wrapper_14105397

book_input

user_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallÉ
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
GPU 2J 8 *,
f'R%
#__inference__wrapped_model_14105197o
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
­
I
-__inference_flatten_35_layer_call_fn_14105512

inputs
identity³
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240`
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
­	
n
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262

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
±	
¬
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105490

inputs,
embedding_lookup_14105484:	N

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’æ
embedding_lookupResourceGatherembedding_lookup_14105484Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14105484*+
_output_shapes
:’’’’’’’’’
*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14105484*+
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
¶!
ø
#__inference__wrapped_model_14105197

user_input

book_inputD
1model_17_user_embedding_embedding_lookup_14105174:	\
D
1model_17_book_embedding_embedding_lookup_14105180:	N

identity¢(model_17/book_embedding/embedding_lookup¢(model_17/user_embedding/embedding_lookupq
model_17/user_embedding/CastCast
user_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’
(model_17/user_embedding/embedding_lookupResourceGather1model_17_user_embedding_embedding_lookup_14105174 model_17/user_embedding/Cast:y:0*
Tindices0*D
_class:
86loc:@model_17/user_embedding/embedding_lookup/14105174*+
_output_shapes
:’’’’’’’’’
*
dtype0ģ
1model_17/user_embedding/embedding_lookup/IdentityIdentity1model_17/user_embedding/embedding_lookup:output:0*
T0*D
_class:
86loc:@model_17/user_embedding/embedding_lookup/14105174*+
_output_shapes
:’’’’’’’’’
±
3model_17/user_embedding/embedding_lookup/Identity_1Identity:model_17/user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
q
model_17/book_embedding/CastCast
book_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’
(model_17/book_embedding/embedding_lookupResourceGather1model_17_book_embedding_embedding_lookup_14105180 model_17/book_embedding/Cast:y:0*
Tindices0*D
_class:
86loc:@model_17/book_embedding/embedding_lookup/14105180*+
_output_shapes
:’’’’’’’’’
*
dtype0ģ
1model_17/book_embedding/embedding_lookup/IdentityIdentity1model_17/book_embedding/embedding_lookup:output:0*
T0*D
_class:
86loc:@model_17/book_embedding/embedding_lookup/14105180*+
_output_shapes
:’’’’’’’’’
±
3model_17/book_embedding/embedding_lookup/Identity_1Identity:model_17/book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
j
model_17/flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   ŗ
model_17/flatten_35/ReshapeReshape<model_17/book_embedding/embedding_lookup/Identity_1:output:0"model_17/flatten_35/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
j
model_17/flatten_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   ŗ
model_17/flatten_34/ReshapeReshape<model_17/user_embedding/embedding_lookup/Identity_1:output:0"model_17/flatten_34/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
`
model_17/dot_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :­
model_17/dot_17/ExpandDims
ExpandDims$model_17/flatten_35/Reshape:output:0'model_17/dot_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
b
 model_17/dot_17/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :±
model_17/dot_17/ExpandDims_1
ExpandDims$model_17/flatten_34/Reshape:output:0)model_17/dot_17/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
©
model_17/dot_17/MatMulBatchMatMulV2#model_17/dot_17/ExpandDims:output:0%model_17/dot_17/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’d
model_17/dot_17/ShapeShapemodel_17/dot_17/MatMul:output:0*
T0*
_output_shapes
:
model_17/dot_17/SqueezeSqueezemodel_17/dot_17/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
o
IdentityIdentity model_17/dot_17/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp)^model_17/book_embedding/embedding_lookup)^model_17/user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’: : 2T
(model_17/book_embedding/embedding_lookup(model_17/book_embedding/embedding_lookup2T
(model_17/user_embedding/embedding_lookup(model_17/user_embedding/embedding_lookup:S O
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
±	
¬
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105507

inputs,
embedding_lookup_14105501:	\

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’æ
embedding_lookupResourceGatherembedding_lookup_14105501Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14105501*+
_output_shapes
:’’’’’’’’’
*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14105501*+
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
Ń
®
+__inference_model_17_layer_call_fn_14105417
inputs_0
inputs_1
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallč
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
GPU 2J 8 *O
fJRH
F__inference_model_17_layer_call_and_return_conditional_losses_14105334o
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
Ó
£
F__inference_model_17_layer_call_and_return_conditional_losses_14105379

user_input

book_input*
user_embedding_14105369:	\
*
book_embedding_14105372:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCallū
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_14105369*
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
GPU 2J 8 *U
fPRN
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216ū
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_14105372*
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
GPU 2J 8 *U
fPRN
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230ē
flatten_35/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240ē
flatten_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248ł
dot_17/PartitionedCallPartitionedCall#flatten_35/PartitionedCall:output:0#flatten_34/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262n
IdentityIdentitydot_17/PartitionedCall:output:0^NoOp*
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
Ą
d
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248

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
±	
¬
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230

inputs,
embedding_lookup_14105224:	N

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’æ
embedding_lookupResourceGatherembedding_lookup_14105224Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14105224*+
_output_shapes
:’’’’’’’’’
*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14105224*+
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
Ń
®
+__inference_model_17_layer_call_fn_14105407
inputs_0
inputs_1
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallč
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
GPU 2J 8 *O
fJRH
F__inference_model_17_layer_call_and_return_conditional_losses_14105265o
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
¢"
Ø
!__inference__traced_save_14105604
file_prefix8
4savev2_book_embedding_embeddings_read_readvariableop8
4savev2_user_embedding_embeddings_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_rmsprop_book_embedding_embeddings_rms_read_readvariableopD
@savev2_rmsprop_user_embedding_embeddings_rms_read_readvariableop
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Æ
value„B¢B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B É
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_book_embedding_embeddings_read_readvariableop4savev2_user_embedding_embeddings_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_rmsprop_book_embedding_embeddings_rms_read_readvariableop@savev2_rmsprop_user_embedding_embeddings_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
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

identity_1Identity_1:output:0*Q
_input_shapes@
>: :	N
:	\
: : : : : : : :	N
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
:

_output_shapes
: 
¢
U
)__inference_dot_17_layer_call_fn_14105535
inputs_0
inputs_1
identity¼
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
GPU 2J 8 *M
fHRF
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262`
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
³

1__inference_user_embedding_layer_call_fn_14105497

inputs
unknown:	\

identity¢StatefulPartitionedCallŲ
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
GPU 2J 8 *U
fPRN
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216s
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
ę
³
F__inference_model_17_layer_call_and_return_conditional_losses_14105473
inputs_0
inputs_1;
(user_embedding_embedding_lookup_14105450:	\
;
(book_embedding_embedding_lookup_14105456:	N

identity¢book_embedding/embedding_lookup¢user_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ū
user_embedding/embedding_lookupResourceGather(user_embedding_embedding_lookup_14105450user_embedding/Cast:y:0*
Tindices0*;
_class1
/-loc:@user_embedding/embedding_lookup/14105450*+
_output_shapes
:’’’’’’’’’
*
dtype0Ń
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@user_embedding/embedding_lookup/14105450*+
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
:’’’’’’’’’ū
book_embedding/embedding_lookupResourceGather(book_embedding_embedding_lookup_14105456book_embedding/Cast:y:0*
Tindices0*;
_class1
/-loc:@book_embedding/embedding_lookup/14105456*+
_output_shapes
:’’’’’’’’’
*
dtype0Ń
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@book_embedding/embedding_lookup/14105456*+
_output_shapes
:’’’’’’’’’

*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
a
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_35/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_35/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
a
flatten_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_34/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_34/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
W
dot_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_17/ExpandDims
ExpandDimsflatten_35/Reshape:output:0dot_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
Y
dot_17/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_17/ExpandDims_1
ExpandDimsflatten_34/Reshape:output:0 dot_17/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’

dot_17/MatMulBatchMatMulV2dot_17/ExpandDims:output:0dot_17/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
dot_17/ShapeShapedot_17/MatMul:output:0*
T0*
_output_shapes
:z
dot_17/SqueezeSqueezedot_17/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
f
IdentityIdentitydot_17/Squeeze:output:0^NoOp*
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
Ą
d
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240

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
µ	
p
D__inference_dot_17_layer_call_and_return_conditional_losses_14105547
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
æ

F__inference_model_17_layer_call_and_return_conditional_losses_14105265

inputs
inputs_1*
user_embedding_14105217:	\
*
book_embedding_14105231:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCall÷
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_14105217*
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
GPU 2J 8 *U
fPRN
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216ł
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_14105231*
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
GPU 2J 8 *U
fPRN
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230ē
flatten_35/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240ē
flatten_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248ł
dot_17/PartitionedCallPartitionedCall#flatten_35/PartitionedCall:output:0#flatten_34/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262n
IdentityIdentitydot_17/PartitionedCall:output:0^NoOp*
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
ę
³
F__inference_model_17_layer_call_and_return_conditional_losses_14105445
inputs_0
inputs_1;
(user_embedding_embedding_lookup_14105422:	\
;
(book_embedding_embedding_lookup_14105428:	N

identity¢book_embedding/embedding_lookup¢user_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ū
user_embedding/embedding_lookupResourceGather(user_embedding_embedding_lookup_14105422user_embedding/Cast:y:0*
Tindices0*;
_class1
/-loc:@user_embedding/embedding_lookup/14105422*+
_output_shapes
:’’’’’’’’’
*
dtype0Ń
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@user_embedding/embedding_lookup/14105422*+
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
:’’’’’’’’’ū
book_embedding/embedding_lookupResourceGather(book_embedding_embedding_lookup_14105428book_embedding/Cast:y:0*
Tindices0*;
_class1
/-loc:@book_embedding/embedding_lookup/14105428*+
_output_shapes
:’’’’’’’’’
*
dtype0Ń
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@book_embedding/embedding_lookup/14105428*+
_output_shapes
:’’’’’’’’’

*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’
a
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_35/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_35/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
a
flatten_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
   
flatten_34/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_34/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
W
dot_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_17/ExpandDims
ExpandDimsflatten_35/Reshape:output:0dot_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
Y
dot_17/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot_17/ExpandDims_1
ExpandDimsflatten_34/Reshape:output:0 dot_17/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’

dot_17/MatMulBatchMatMulV2dot_17/ExpandDims:output:0dot_17/ExpandDims_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
dot_17/ShapeShapedot_17/MatMul:output:0*
T0*
_output_shapes
:z
dot_17/SqueezeSqueezedot_17/MatMul:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
f
IdentityIdentitydot_17/Squeeze:output:0^NoOp*
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
ė/
ų
$__inference__traced_restore_14105647
file_prefix=
*assignvariableop_book_embedding_embeddings:	N
?
,assignvariableop_1_user_embedding_embeddings:	\
)
assignvariableop_2_rmsprop_iter:	 *
 assignvariableop_3_rmsprop_decay: 2
(assignvariableop_4_rmsprop_learning_rate: -
#assignvariableop_5_rmsprop_momentum: (
assignvariableop_6_rmsprop_rho: "
assignvariableop_7_total: "
assignvariableop_8_count: K
8assignvariableop_9_rmsprop_book_embedding_embeddings_rms:	N
L
9assignvariableop_10_rmsprop_user_embedding_embeddings_rms:	\

identity_12¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Æ
value„B¢B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B Ś
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	[
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
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
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
:§
AssignVariableOp_9AssignVariableOp8assignvariableop_9_rmsprop_book_embedding_embeddings_rmsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_10AssignVariableOp9assignvariableop_10_rmsprop_user_embedding_embeddings_rmsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Į
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: ®
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
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
±	
¬
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216

inputs,
embedding_lookup_14105210:	\

identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’æ
embedding_lookupResourceGatherembedding_lookup_14105210Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14105210*+
_output_shapes
:’’’’’’’’’
*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14105210*+
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
³

1__inference_book_embedding_layer_call_fn_14105480

inputs
unknown:	N

identity¢StatefulPartitionedCallŲ
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
GPU 2J 8 *U
fPRN
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230s
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
Ą
d
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105518

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
Ż
²
+__inference_model_17_layer_call_fn_14105351

user_input

book_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallģ
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
GPU 2J 8 *O
fJRH
F__inference_model_17_layer_call_and_return_conditional_losses_14105334o
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
Ą
d
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105529

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
­
I
-__inference_flatten_34_layer_call_fn_14105523

inputs
identity³
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248`
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
Ż
²
+__inference_model_17_layer_call_fn_14105272

user_input

book_input
unknown:	\

	unknown_0:	N

identity¢StatefulPartitionedCallģ
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
GPU 2J 8 *O
fJRH
F__inference_model_17_layer_call_and_return_conditional_losses_14105265o
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
æ

F__inference_model_17_layer_call_and_return_conditional_losses_14105334

inputs
inputs_1*
user_embedding_14105324:	\
*
book_embedding_14105327:	N

identity¢&book_embedding/StatefulPartitionedCall¢&user_embedding/StatefulPartitionedCall÷
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_14105324*
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
GPU 2J 8 *U
fPRN
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105216ł
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_14105327*
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
GPU 2J 8 *U
fPRN
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105230ē
flatten_35/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105240ē
flatten_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *Q
fLRJ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105248ł
dot_17/PartitionedCallPartitionedCall#flatten_35/PartitionedCall:output:0#flatten_34/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dot_17_layer_call_and_return_conditional_losses_14105262n
IdentityIdentitydot_17/PartitionedCall:output:0^NoOp*
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
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ņ
serving_defaultŽ
A

book_input3
serving_default_book_input:0’’’’’’’’’
A

user_input3
serving_default_user_input:0’’’’’’’’’:
dot_170
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:É
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
į
7trace_0
8trace_1
9trace_2
:trace_32ö
+__inference_model_17_layer_call_fn_14105272
+__inference_model_17_layer_call_fn_14105407
+__inference_model_17_layer_call_fn_14105417
+__inference_model_17_layer_call_fn_14105351æ
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
Ķ
;trace_0
<trace_1
=trace_2
>trace_32ā
F__inference_model_17_layer_call_and_return_conditional_losses_14105445
F__inference_model_17_layer_call_and_return_conditional_losses_14105473
F__inference_model_17_layer_call_and_return_conditional_losses_14105365
F__inference_model_17_layer_call_and_return_conditional_losses_14105379æ
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
ŻBŚ
#__inference__wrapped_model_14105197
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
h
?iter
	@decay
Alearning_rate
Bmomentum
Crho	rmsm	rmsn"
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
õ
Jtrace_02Ų
1__inference_book_embedding_layer_call_fn_14105480¢
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

Ktrace_02ó
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105490¢
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
õ
Qtrace_02Ų
1__inference_user_embedding_layer_call_fn_14105497¢
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

Rtrace_02ó
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105507¢
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
ń
Xtrace_02Ō
-__inference_flatten_35_layer_call_fn_14105512¢
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

Ytrace_02ļ
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105518¢
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
ń
_trace_02Ō
-__inference_flatten_34_layer_call_fn_14105523¢
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

`trace_02ļ
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105529¢
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
ķ
ftrace_02Š
)__inference_dot_17_layer_call_fn_14105535¢
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

gtrace_02ė
D__inference_dot_17_layer_call_and_return_conditional_losses_14105547¢
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
B
+__inference_model_17_layer_call_fn_14105272
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
B
+__inference_model_17_layer_call_fn_14105407inputs/0inputs/1"æ
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
B
+__inference_model_17_layer_call_fn_14105417inputs/0inputs/1"æ
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
B
+__inference_model_17_layer_call_fn_14105351
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
£B 
F__inference_model_17_layer_call_and_return_conditional_losses_14105445inputs/0inputs/1"æ
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
£B 
F__inference_model_17_layer_call_and_return_conditional_losses_14105473inputs/0inputs/1"æ
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
§B¤
F__inference_model_17_layer_call_and_return_conditional_losses_14105365
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
§B¤
F__inference_model_17_layer_call_and_return_conditional_losses_14105379
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
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
ŚB×
&__inference_signature_wrapper_14105397
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
åBā
1__inference_book_embedding_layer_call_fn_14105480inputs"¢
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
Bż
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105490inputs"¢
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
åBā
1__inference_user_embedding_layer_call_fn_14105497inputs"¢
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
Bż
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105507inputs"¢
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
įBŽ
-__inference_flatten_35_layer_call_fn_14105512inputs"¢
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
üBł
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105518inputs"¢
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
įBŽ
-__inference_flatten_34_layer_call_fn_14105523inputs"¢
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
üBł
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105529inputs"¢
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
éBę
)__inference_dot_17_layer_call_fn_14105535inputs/0inputs/1"¢
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
B
D__inference_dot_17_layer_call_and_return_conditional_losses_14105547inputs/0inputs/1"¢
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
6:4	N
2%RMSprop/book_embedding/embeddings/rms
6:4	\
2%RMSprop/user_embedding/embeddings/rms½
#__inference__wrapped_model_14105197^¢[
T¢Q
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
Ŗ "/Ŗ,
*
dot_17 
dot_17’’’’’’’’’Æ
L__inference_book_embedding_layer_call_and_return_conditional_losses_14105490_/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’

 
1__inference_book_embedding_layer_call_fn_14105480R/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
Ģ
D__inference_dot_17_layer_call_and_return_conditional_losses_14105547Z¢W
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
 £
)__inference_dot_17_layer_call_fn_14105535vZ¢W
P¢M
KH
"
inputs/0’’’’’’’’’

"
inputs/1’’’’’’’’’

Ŗ "’’’’’’’’’Ø
H__inference_flatten_34_layer_call_and_return_conditional_losses_14105529\3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 
-__inference_flatten_34_layer_call_fn_14105523O3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
Ø
H__inference_flatten_35_layer_call_and_return_conditional_losses_14105518\3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 
-__inference_flatten_35_layer_call_fn_14105512O3¢0
)¢&
$!
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
Ž
F__inference_model_17_layer_call_and_return_conditional_losses_14105365f¢c
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
 Ž
F__inference_model_17_layer_call_and_return_conditional_losses_14105379f¢c
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
 Ś
F__inference_model_17_layer_call_and_return_conditional_losses_14105445b¢_
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
 Ś
F__inference_model_17_layer_call_and_return_conditional_losses_14105473b¢_
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
 ¶
+__inference_model_17_layer_call_fn_14105272f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’¶
+__inference_model_17_layer_call_fn_14105351f¢c
\¢Y
OL
$!

user_input’’’’’’’’’
$!

book_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’²
+__inference_model_17_layer_call_fn_14105407b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’²
+__inference_model_17_layer_call_fn_14105417b¢_
X¢U
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
p

 
Ŗ "’’’’’’’’’×
&__inference_signature_wrapper_14105397¬u¢r
¢ 
kŖh
2

book_input$!

book_input’’’’’’’’’
2

user_input$!

user_input’’’’’’’’’"/Ŗ,
*
dot_17 
dot_17’’’’’’’’’Æ
L__inference_user_embedding_layer_call_and_return_conditional_losses_14105507_/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’

 
1__inference_user_embedding_layer_call_fn_14105497R/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
