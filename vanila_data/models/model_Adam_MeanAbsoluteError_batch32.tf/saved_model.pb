▄Ю
т╚
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
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
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ц
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	ѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8ЇЧ
Ю
 Adam/user_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ј\
*1
shared_name" Adam/user_embedding/embeddings/v
ќ
4Adam/user_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/v*
_output_shapes
:	Ј\
*
dtype0
Ю
 Adam/book_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љN
*1
shared_name" Adam/book_embedding/embeddings/v
ќ
4Adam/book_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/book_embedding/embeddings/v*
_output_shapes
:	љN
*
dtype0
Ю
 Adam/user_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ј\
*1
shared_name" Adam/user_embedding/embeddings/m
ќ
4Adam/user_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/m*
_output_shapes
:	Ј\
*
dtype0
Ю
 Adam/book_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љN
*1
shared_name" Adam/book_embedding/embeddings/m
ќ
4Adam/book_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/book_embedding/embeddings/m*
_output_shapes
:	љN
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
Ј
user_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ј\
**
shared_nameuser_embedding/embeddings
ѕ
-user_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_embedding/embeddings*
_output_shapes
:	Ј\
*
dtype0
Ј
book_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љN
**
shared_namebook_embedding/embeddings
ѕ
-book_embedding/embeddings/Read/ReadVariableOpReadVariableOpbook_embedding/embeddings*
_output_shapes
:	љN
*
dtype0
}
serving_default_book_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_user_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_book_inputserving_default_user_inputuser_embedding/embeddingsbook_embedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference_signature_wrapper_2344287

NoOpNoOp
Э(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*│(
valueЕ(Bд( BЪ(
█
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
а
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
а
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
ј
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
ј
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
ј
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
░
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
Њ
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
Њ
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
Љ
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
Љ
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
Љ
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
Љі
VARIABLE_VALUE Adam/book_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE Adam/user_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE Adam/book_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE Adam/user_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Й
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
GPU 2J 8ѓ *)
f$R"
 __inference__traced_save_2344500
х
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
GPU 2J 8ѓ *,
f'R%
#__inference__traced_restore_2344549Ве
р 
▒
"__inference__wrapped_model_2344087

user_input

book_inputB
/model_4_user_embedding_embedding_lookup_2344064:	Ј\
B
/model_4_book_embedding_embedding_lookup_2344070:	љN

identityѕб'model_4/book_embedding/embedding_lookupб'model_4/user_embedding/embedding_lookupp
model_4/user_embedding/CastCast
user_input*

DstT0*

SrcT0*'
_output_shapes
:         Ў
'model_4/user_embedding/embedding_lookupResourceGather/model_4_user_embedding_embedding_lookup_2344064model_4/user_embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@model_4/user_embedding/embedding_lookup/2344064*+
_output_shapes
:         
*
dtype0У
0model_4/user_embedding/embedding_lookup/IdentityIdentity0model_4/user_embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@model_4/user_embedding/embedding_lookup/2344064*+
_output_shapes
:         
»
2model_4/user_embedding/embedding_lookup/Identity_1Identity9model_4/user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
p
model_4/book_embedding/CastCast
book_input*

DstT0*

SrcT0*'
_output_shapes
:         Ў
'model_4/book_embedding/embedding_lookupResourceGather/model_4_book_embedding_embedding_lookup_2344070model_4/book_embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@model_4/book_embedding/embedding_lookup/2344070*+
_output_shapes
:         
*
dtype0У
0model_4/book_embedding/embedding_lookup/IdentityIdentity0model_4/book_embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@model_4/book_embedding/embedding_lookup/2344070*+
_output_shapes
:         
»
2model_4/book_embedding/embedding_lookup/Identity_1Identity9model_4/book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
h
model_4/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   х
model_4/flatten_9/ReshapeReshape;model_4/book_embedding/embedding_lookup/Identity_1:output:0 model_4/flatten_9/Const:output:0*
T0*'
_output_shapes
:         
h
model_4/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   х
model_4/flatten_8/ReshapeReshape;model_4/user_embedding/embedding_lookup/Identity_1:output:0 model_4/flatten_8/Const:output:0*
T0*'
_output_shapes
:         
^
model_4/dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Д
model_4/dot_4/ExpandDims
ExpandDims"model_4/flatten_9/Reshape:output:0%model_4/dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         
`
model_4/dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
model_4/dot_4/ExpandDims_1
ExpandDims"model_4/flatten_8/Reshape:output:0'model_4/dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         
Б
model_4/dot_4/MatMulBatchMatMulV2!model_4/dot_4/ExpandDims:output:0#model_4/dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:         `
model_4/dot_4/ShapeShapemodel_4/dot_4/MatMul:output:0*
T0*
_output_shapes
:ѕ
model_4/dot_4/SqueezeSqueezemodel_4/dot_4/MatMul:output:0*
T0*'
_output_shapes
:         *
squeeze_dims
m
IdentityIdentitymodel_4/dot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         џ
NoOpNoOp(^model_4/book_embedding/embedding_lookup(^model_4/user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2R
'model_4/book_embedding/embedding_lookup'model_4/book_embedding/embedding_lookup2R
'model_4/user_embedding/embedding_lookup'model_4/user_embedding/embedding_lookup:S O
'
_output_shapes
:         
$
_user_specified_name
user_input:SO
'
_output_shapes
:         
$
_user_specified_name
book_input
Й
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344419

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Е
G
+__inference_flatten_9_layer_call_fn_2344402

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▒
Ё
0__inference_book_embedding_layer_call_fn_2344370

inputs
unknown:	љN

identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│
г
%__inference_signature_wrapper_2344287

book_input

user_input
unknown:	Ј\

	unknown_0:	љN

identityѕбStatefulPartitionedCall╚
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__wrapped_model_2344087o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
book_input:SO
'
_output_shapes
:         
$
_user_specified_name
user_input
═
г
)__inference_model_4_layer_call_fn_2344297
inputs_0
inputs_1
unknown:	Ј\

	unknown_0:	љN

identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_2344155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
к
»
D__inference_model_4_layer_call_and_return_conditional_losses_2344363
inputs_0
inputs_1:
'user_embedding_embedding_lookup_2344340:	Ј\
:
'book_embedding_embedding_lookup_2344346:	љN

identityѕбbook_embedding/embedding_lookupбuser_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:         щ
user_embedding/embedding_lookupResourceGather'user_embedding_embedding_lookup_2344340user_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@user_embedding/embedding_lookup/2344340*+
_output_shapes
:         
*
dtype0л
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@user_embedding/embedding_lookup/2344340*+
_output_shapes
:         
Ъ
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
f
book_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:         щ
book_embedding/embedding_lookupResourceGather'book_embedding_embedding_lookup_2344346book_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@book_embedding/embedding_lookup/2344346*+
_output_shapes
:         
*
dtype0л
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@book_embedding/embedding_lookup/2344346*+
_output_shapes
:         
Ъ
*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   Ю
flatten_9/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:         
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   Ю
flatten_8/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:         
V
dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ј
dot_4/ExpandDims
ExpandDimsflatten_9/Reshape:output:0dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         
X
dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
dot_4/ExpandDims_1
ExpandDimsflatten_8/Reshape:output:0dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         
І
dot_4/MatMulBatchMatMulV2dot_4/ExpandDims:output:0dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:         P
dot_4/ShapeShapedot_4/MatMul:output:0*
T0*
_output_shapes
:x
dot_4/SqueezeSqueezedot_4/MatMul:output:0*
T0*'
_output_shapes
:         *
squeeze_dims
e
IdentityIdentitydot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         і
NoOpNoOp ^book_embedding/embedding_lookup ^user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2B
book_embedding/embedding_lookupbook_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┘
░
)__inference_model_4_layer_call_fn_2344241

user_input

book_input
unknown:	Ј\

	unknown_0:	љN

identityѕбStatefulPartitionedCallЖ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_2344224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
user_input:SO
'
_output_shapes
:         
$
_user_specified_name
book_input
Ф
Ў
D__inference_model_4_layer_call_and_return_conditional_losses_2344155

inputs
inputs_1)
user_embedding_2344107:	Ј\
)
book_embedding_2344121:	љN

identityѕб&book_embedding/StatefulPartitionedCallб&user_embedding/StatefulPartitionedCallш
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_2344107*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106э
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_2344121*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120С
flatten_9/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130С
flatten_8/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138З
dot_4/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
г	
ф
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344397

inputs+
embedding_lookup_2344391:	Ј\

identityѕбembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         й
embedding_lookupResourceGatherembedding_lookup_2344391Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2344391*+
_output_shapes
:         
*
dtype0Б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2344391*+
_output_shapes
:         
Ђ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┐
Ъ
D__inference_model_4_layer_call_and_return_conditional_losses_2344269

user_input

book_input)
user_embedding_2344259:	Ј\
)
book_embedding_2344262:	љN

identityѕб&book_embedding/StatefulPartitionedCallб&user_embedding/StatefulPartitionedCallщ
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_2344259*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106щ
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_2344262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120С
flatten_9/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130С
flatten_8/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138З
dot_4/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
user_input:SO
'
_output_shapes
:         
$
_user_specified_name
book_input
┐
Ъ
D__inference_model_4_layer_call_and_return_conditional_losses_2344255

user_input

book_input)
user_embedding_2344245:	Ј\
)
book_embedding_2344248:	љN

identityѕб&book_embedding/StatefulPartitionedCallб&user_embedding/StatefulPartitionedCallщ
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_2344245*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106щ
&book_embedding/StatefulPartitionedCallStatefulPartitionedCall
book_inputbook_embedding_2344248*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120С
flatten_9/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130С
flatten_8/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138З
dot_4/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
user_input:SO
'
_output_shapes
:         
$
_user_specified_name
book_input
═
г
)__inference_model_4_layer_call_fn_2344307
inputs_0
inputs_1
unknown:	Ј\

	unknown_0:	љN

identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_2344224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
г	
ф
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344380

inputs+
embedding_lookup_2344374:	љN

identityѕбembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         й
embedding_lookupResourceGatherembedding_lookup_2344374Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2344374*+
_output_shapes
:         
*
dtype0Б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2344374*+
_output_shapes
:         
Ђ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
░
)__inference_model_4_layer_call_fn_2344162

user_input

book_input
unknown:	Ј\

	unknown_0:	љN

identityѕбStatefulPartitionedCallЖ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_2344155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
user_input:SO
'
_output_shapes
:         
$
_user_specified_name
book_input
ё&
Љ
 __inference__traced_save_2344500
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

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: │
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▄
valueмB¤B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B Г
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_book_embedding_embeddings_read_readvariableop4savev2_user_embedding_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_book_embedding_embeddings_m_read_readvariableop;savev2_adam_user_embedding_embeddings_m_read_readvariableop;savev2_adam_book_embedding_embeddings_v_read_readvariableop;savev2_adam_user_embedding_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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
T: :	љN
:	Ј\
: : : : : : : :	љN
:	Ј\
:	љN
:	Ј\
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	љN
:%!

_output_shapes
:	Ј\
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
:	љN
:%!

_output_shapes
:	Ј\
:%!

_output_shapes
:	љN
:%!

_output_shapes
:	Ј\
:

_output_shapes
: 
▒
Ё
0__inference_user_embedding_layer_call_fn_2344387

inputs
unknown:	Ј\

identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ф	
l
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152

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
:         
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
:         
y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:         D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:         *
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
щ7
Ю
#__inference__traced_restore_2344549
file_prefix=
*assignvariableop_book_embedding_embeddings:	љN
?
,assignvariableop_1_user_embedding_embeddings:	Ј\
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: F
3assignvariableop_9_adam_book_embedding_embeddings_m:	љN
G
4assignvariableop_10_adam_user_embedding_embeddings_m:	Ј\
G
4assignvariableop_11_adam_book_embedding_embeddings_v:	љN
G
4assignvariableop_12_adam_user_embedding_embeddings_v:	Ј\

identity_14ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Х
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▄
valueмB¤B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHї
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOpAssignVariableOp*assignvariableop_book_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_1AssignVariableOp,assignvariableop_1_user_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:І
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_9AssignVariableOp3assignvariableop_9_adam_book_embedding_embeddings_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_10AssignVariableOp4assignvariableop_10_adam_user_embedding_embeddings_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_book_embedding_embeddings_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_user_embedding_embeddings_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ь
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: ┌
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
Ф
Ў
D__inference_model_4_layer_call_and_return_conditional_losses_2344224

inputs
inputs_1)
user_embedding_2344214:	Ј\
)
book_embedding_2344217:	љN

identityѕб&book_embedding/StatefulPartitionedCallб&user_embedding/StatefulPartitionedCallш
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_2344214*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106э
&book_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1book_embedding_2344217*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120С
flatten_9/PartitionedCallPartitionedCall/book_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130С
flatten_8/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138З
dot_4/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp'^book_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2P
&book_embedding/StatefulPartitionedCall&book_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
G
+__inference_flatten_8_layer_call_fn_2344413

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ъ
S
'__inference_dot_4_layer_call_fn_2344425
inputs_0
inputs_1
identity║
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dot_4_layer_call_and_return_conditional_losses_2344152`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
г	
ф
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344120

inputs+
embedding_lookup_2344114:	љN

identityѕбembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         й
embedding_lookupResourceGatherembedding_lookup_2344114Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2344114*+
_output_shapes
:         
*
dtype0Б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2344114*+
_output_shapes
:         
Ђ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
г	
ф
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344106

inputs+
embedding_lookup_2344100:	Ј\

identityѕбembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         й
embedding_lookupResourceGatherembedding_lookup_2344100Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2344100*+
_output_shapes
:         
*
dtype0Б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2344100*+
_output_shapes
:         
Ђ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│	
n
B__inference_dot_4_layer_call_and_return_conditional_losses_2344437
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
:         
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
:         
y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:         D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:         *
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
Й
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344408

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Й
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344138

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
к
»
D__inference_model_4_layer_call_and_return_conditional_losses_2344335
inputs_0
inputs_1:
'user_embedding_embedding_lookup_2344312:	Ј\
:
'book_embedding_embedding_lookup_2344318:	љN

identityѕбbook_embedding/embedding_lookupбuser_embedding/embedding_lookupf
user_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:         щ
user_embedding/embedding_lookupResourceGather'user_embedding_embedding_lookup_2344312user_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@user_embedding/embedding_lookup/2344312*+
_output_shapes
:         
*
dtype0л
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@user_embedding/embedding_lookup/2344312*+
_output_shapes
:         
Ъ
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
f
book_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:         щ
book_embedding/embedding_lookupResourceGather'book_embedding_embedding_lookup_2344318book_embedding/Cast:y:0*
Tindices0*:
_class0
.,loc:@book_embedding/embedding_lookup/2344318*+
_output_shapes
:         
*
dtype0л
(book_embedding/embedding_lookup/IdentityIdentity(book_embedding/embedding_lookup:output:0*
T0*:
_class0
.,loc:@book_embedding/embedding_lookup/2344318*+
_output_shapes
:         
Ъ
*book_embedding/embedding_lookup/Identity_1Identity1book_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         
`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   Ю
flatten_9/ReshapeReshape3book_embedding/embedding_lookup/Identity_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:         
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   Ю
flatten_8/ReshapeReshape3user_embedding/embedding_lookup/Identity_1:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:         
V
dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ј
dot_4/ExpandDims
ExpandDimsflatten_9/Reshape:output:0dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         
X
dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
dot_4/ExpandDims_1
ExpandDimsflatten_8/Reshape:output:0dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         
І
dot_4/MatMulBatchMatMulV2dot_4/ExpandDims:output:0dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:         P
dot_4/ShapeShapedot_4/MatMul:output:0*
T0*
_output_shapes
:x
dot_4/SqueezeSqueezedot_4/MatMul:output:0*
T0*'
_output_shapes
:         *
squeeze_dims
e
IdentityIdentitydot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         і
NoOpNoOp ^book_embedding/embedding_lookup ^user_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:         :         : : 2B
book_embedding/embedding_lookupbook_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Й
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344130

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ы
serving_defaultП
A

book_input3
serving_default_book_input:0         
A

user_input3
serving_default_user_input:0         9
dot_40
StatefulPartitionedCall:0         tensorflow/serving/predict:╝Ї
Ы
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
х
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
х
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Ц
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
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
╩
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
┘
7trace_0
8trace_1
9trace_2
:trace_32Ь
)__inference_model_4_layer_call_fn_2344162
)__inference_model_4_layer_call_fn_2344297
)__inference_model_4_layer_call_fn_2344307
)__inference_model_4_layer_call_fn_2344241┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z7trace_0z8trace_1z9trace_2z:trace_3
┼
;trace_0
<trace_1
=trace_2
>trace_32┌
D__inference_model_4_layer_call_and_return_conditional_losses_2344335
D__inference_model_4_layer_call_and_return_conditional_losses_2344363
D__inference_model_4_layer_call_and_return_conditional_losses_2344255
D__inference_model_4_layer_call_and_return_conditional_losses_2344269┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z;trace_0z<trace_1z=trace_2z>trace_3
▄B┘
"__inference__wrapped_model_2344087
user_input
book_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
З
Jtrace_02О
0__inference_book_embedding_layer_call_fn_2344370б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zJtrace_0
Ј
Ktrace_02Ы
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344380б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zKtrace_0
,:*	љN
2book_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
З
Qtrace_02О
0__inference_user_embedding_layer_call_fn_2344387б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zQtrace_0
Ј
Rtrace_02Ы
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344397б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zRtrace_0
,:*	Ј\
2user_embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
№
Xtrace_02м
+__inference_flatten_9_layer_call_fn_2344402б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zXtrace_0
і
Ytrace_02ь
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344408б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zYtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
№
_trace_02м
+__inference_flatten_8_layer_call_fn_2344413б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z_trace_0
і
`trace_02ь
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344419б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z`trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
в
ftrace_02╬
'__inference_dot_4_layer_call_fn_2344425б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zftrace_0
є
gtrace_02ж
B__inference_dot_4_layer_call_and_return_conditional_losses_2344437б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
іBЄ
)__inference_model_4_layer_call_fn_2344162
user_input
book_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
)__inference_model_4_layer_call_fn_2344297inputs/0inputs/1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
)__inference_model_4_layer_call_fn_2344307inputs/0inputs/1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
)__inference_model_4_layer_call_fn_2344241
user_input
book_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
D__inference_model_4_layer_call_and_return_conditional_losses_2344335inputs/0inputs/1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
D__inference_model_4_layer_call_and_return_conditional_losses_2344363inputs/0inputs/1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЦBб
D__inference_model_4_layer_call_and_return_conditional_losses_2344255
user_input
book_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЦBб
D__inference_model_4_layer_call_and_return_conditional_losses_2344269
user_input
book_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
┘Bо
%__inference_signature_wrapper_2344287
book_input
user_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_book_embedding_layer_call_fn_2344370inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344380inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_user_embedding_layer_call_fn_2344387inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344397inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▀B▄
+__inference_flatten_9_layer_call_fn_2344402inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344408inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▀B▄
+__inference_flatten_8_layer_call_fn_2344413inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344419inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
'__inference_dot_4_layer_call_fn_2344425inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
B__inference_dot_4_layer_call_and_return_conditional_losses_2344437inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
1:/	љN
2 Adam/book_embedding/embeddings/m
1:/	Ј\
2 Adam/user_embedding/embeddings/m
1:/	љN
2 Adam/book_embedding/embeddings/v
1:/	Ј\
2 Adam/user_embedding/embeddings/v║
"__inference__wrapped_model_2344087Њ^б[
TбQ
OџL
$і!

user_input         
$і!

book_input         
ф "-ф*
(
dot_4і
dot_4         «
K__inference_book_embedding_layer_call_and_return_conditional_losses_2344380_/б,
%б"
 і
inputs         
ф ")б&
і
0         

џ є
0__inference_book_embedding_layer_call_fn_2344370R/б,
%б"
 і
inputs         
ф "і         
╩
B__inference_dot_4_layer_call_and_return_conditional_losses_2344437ЃZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         

ф "%б"
і
0         
џ А
'__inference_dot_4_layer_call_fn_2344425vZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         

ф "і         д
F__inference_flatten_8_layer_call_and_return_conditional_losses_2344419\3б0
)б&
$і!
inputs         

ф "%б"
і
0         

џ ~
+__inference_flatten_8_layer_call_fn_2344413O3б0
)б&
$і!
inputs         

ф "і         
д
F__inference_flatten_9_layer_call_and_return_conditional_losses_2344408\3б0
)б&
$і!
inputs         

ф "%б"
і
0         

џ ~
+__inference_flatten_9_layer_call_fn_2344402O3б0
)б&
$і!
inputs         

ф "і         
▄
D__inference_model_4_layer_call_and_return_conditional_losses_2344255Њfбc
\бY
OџL
$і!

user_input         
$і!

book_input         
p 

 
ф "%б"
і
0         
џ ▄
D__inference_model_4_layer_call_and_return_conditional_losses_2344269Њfбc
\бY
OџL
$і!

user_input         
$і!

book_input         
p

 
ф "%б"
і
0         
џ п
D__inference_model_4_layer_call_and_return_conditional_losses_2344335Јbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         
p 

 
ф "%б"
і
0         
џ п
D__inference_model_4_layer_call_and_return_conditional_losses_2344363Јbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         
p

 
ф "%б"
і
0         
џ ┤
)__inference_model_4_layer_call_fn_2344162єfбc
\бY
OџL
$і!

user_input         
$і!

book_input         
p 

 
ф "і         ┤
)__inference_model_4_layer_call_fn_2344241єfбc
\бY
OџL
$і!

user_input         
$і!

book_input         
p

 
ф "і         ░
)__inference_model_4_layer_call_fn_2344297ѓbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         
p 

 
ф "і         ░
)__inference_model_4_layer_call_fn_2344307ѓbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         
p

 
ф "і         н
%__inference_signature_wrapper_2344287фuбr
б 
kфh
2

book_input$і!

book_input         
2

user_input$і!

user_input         "-ф*
(
dot_4і
dot_4         «
K__inference_user_embedding_layer_call_and_return_conditional_losses_2344397_/б,
%б"
 і
inputs         
ф ")б&
і
0         

џ є
0__inference_user_embedding_layer_call_fn_2344387R/б,
%б"
 і
inputs         
ф "і         
