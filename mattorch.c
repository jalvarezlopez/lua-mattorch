/*
  + This is a wrapper for matlab std I/O functions

  + Supported Types (LOAD):
        mxCELL_CLASS      Y (Only read 1-dim cells. If dim. is more than 2, it force to read them as 1-dim.)
        mxSTRUCT_CLASS    Y
        mxLOGICAL_CLASS
        mxCHAR_CLASS      Y
        mxDOUBLE_CLASS    Y
        mxSINGLE_CLASS    Y
        mxINT8_CLASS      Y
        mxUINT8_CLASS     Y
        mxINT16_CLASS     Y
        mxUINT16_CLASS    Y (casts to INT16)
        mxINT32_CLASS     Y
        mxUINT32_CLASS    Y (casts to INT32)
        mxINT64_CLASS
        mxUINT64_CLASS
        mxFUNCTION_CLASS

  + Supported Types (SAVE):
        LUA_TNIL            -> mxINT32_CLASS
        LUA_TBOOLEAN        -> mxINT8_CLASS
        LUA_TNUMBER         -> mxDOUBLE_CLASS
        LUA_TSTRING         -> mxCHAR_CLASS
        torch.DoubleTensor  -> mxDOUBLE_CLASS
        torch.FloatTensor   -> mxSINGLE_CLASS
        LUA_TTABLE          -> mxSTRUCT_CLASS
        LUA_TTABLE{array}   -> mxCELL_CLASS
*/

// To load this lib in LUA:
// require 'libmatlab'

#include <luaT.h>
#include <TH/TH.h>

#include "mat.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct S_ARRAY_ITEM{
	struct S_ARRAY_ITEM* perv;
	struct S_ARRAY_ITEM* next;
	uint32_t idx;
	mxArray* value;
} T_ARRAY_ITEM;

typedef  struct S_ARRAY{
	struct S_ARRAY_ITEM* first;
	struct S_ARRAY_ITEM* last;
	uint32_t numOfItem;
	uint32_t maxIdxValue;
}T_ARRAY;

static void readAndPushMxArray(lua_State *L, const mxArray* src);
static void pushMxCellData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims);
static void pushMxStructData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims);
static mxArray* assignData(lua_State *L, int pos);

//---------------- functions for T_ARRAY & T_ARRAY_ITEM --------------------
T_ARRAY_ITEM* constractT_ARRAY_ITEM(void);
void distoryT_ARRAY_ITEM(T_ARRAY_ITEM* trg);
T_ARRAY* constractT_ARRAY(void);
void distoryT_ARRAY(T_ARRAY* trg);
void addT_ARRAY_item(T_ARRAY* trg, uint32_t idx, mxArray* value);

T_ARRAY_ITEM* constractT_ARRAY_ITEM(void)
{
	T_ARRAY_ITEM* trg = (T_ARRAY_ITEM*)malloc(sizeof(T_ARRAY_ITEM));
	trg->idx = 0;
	trg->value = NULL;
	trg->perv = NULL;
	trg->next = NULL;
	return trg;
}
void distoryT_ARRAY_ITEM(T_ARRAY_ITEM* trg)
{
	if(trg->perv != NULL)
		trg->perv->next = NULL;
	if(trg->next != NULL)
		trg->next->perv = NULL;
	if(trg->value != NULL)
		mxDestroyArray(trg->value);
	free(trg);
}
T_ARRAY* constractT_ARRAY(void)
{
	T_ARRAY* trg = (T_ARRAY*)malloc(sizeof(T_ARRAY));
	trg->first = NULL;
	trg->last = NULL;
	trg->numOfItem = 0;
	trg->maxIdxValue = 0;
	return trg;
}
void distoryT_ARRAY(T_ARRAY* trg)
{
	T_ARRAY_ITEM* next = NULL;
	T_ARRAY_ITEM* curr = trg->first;
	while(curr != NULL)
	{
		next = curr->next;
		distoryT_ARRAY_ITEM(curr);		
		curr = next;
	}
	free(trg);
}
void addT_ARRAY_item(T_ARRAY* trg, uint32_t idx, mxArray* value)
{
	T_ARRAY_ITEM* last = trg->last;
	T_ARRAY_ITEM* curr = constractT_ARRAY_ITEM();
// 	printf("addT_ARRAY_item @ par:%p, last:%p, curr:%p\n", trg, last, curr);
	curr->idx = idx;
	curr->value = value;
	curr->perv = last;
	trg->last = curr;
	if(last != NULL)
		last->next = curr;
	if(trg->first == NULL)
		trg->first = curr;
	if(trg->maxIdxValue < curr->idx)
		trg->maxIdxValue = curr->idx;
	(trg->numOfItem)++;
// 	printf(" >> first: %p last :%p maxIdxValue : %u, numOfItem : %u\n", trg->first,
// 	trg->last, trg->maxIdxValue, trg->numOfItem);
}

//-----------------------------------------------------------

static void stackDump (lua_State *L, const char* prefix) {
	int i;
	int top = lua_gettop(L);
	printf("----------------- stack --------------------------\n %s> ",prefix);
	for (i = 1; i <= top; i++) {  /* repeat for each level */
		int t = lua_type(L, i);
		switch (t) {

			case LUA_TSTRING:  /* strings */
			printf("`%s'", lua_tostring(L, i));
			break;

			case LUA_TBOOLEAN:  /* booleans */
			printf(lua_toboolean(L, i) ? "true" : "false");
			break;

			case LUA_TNUMBER:  /* numbers */
			printf("%g", lua_tonumber(L, i));
			break;

			default:  /* other values */
			printf("%s", lua_typename(L, t));
			break;

		}
		printf("  ");  /* put a separator */
	}
	printf("\n");  /* end the listing */
	printf("--------------------------------------------------\n");
}
    

void pushMxStructData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims)
{
    mwSize numElements = mxGetNumberOfElements(src);
    mwIndex index;
    int numFields, fidx;
    const char  *fname;
    const mxArray *field_array_ptr;    
    numFields = mxGetNumberOfFields(src);
    
    lua_newtable(L);
    for(fidx=0; fidx<numFields; fidx++)
    {
        fname = mxGetFieldNameByNumber(src, fidx);    
        lua_pushstring(L, fname);    // set field name as a key
        if(numElements < 1)
            lua_pushstring(L, "NULL");
        else if(numElements == 1)
        {
            field_array_ptr = mxGetFieldByNumber(src, 0, fidx);
            if(field_array_ptr == NULL)
                lua_pushstring(L, "NULL");
            else
                readAndPushMxArray(L, field_array_ptr);
        }else{
            lua_newtable(L);
            lua_pushstring(L, "Length");
            lua_pushinteger(L, numElements);
            lua_settable(L, -3);
            for(index=0; index<numElements;    index++)
            {
                lua_pushinteger(L, (index+1));
                field_array_ptr = mxGetFieldByNumber(src, index, fidx);
                if(field_array_ptr == NULL)
                    lua_pushstring(L, "NULL");
                else
                readAndPushMxArray(L, field_array_ptr);
                lua_settable(L, -3);
            }
        }        
        lua_settable(L, -3);
    }
}

void pushMxCellData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims)
{
    mwIndex index;
    mwSize numElements = mxGetNumberOfElements(src);

    // Create sub-table and put Length
    lua_newtable(L);
    lua_pushstring(L, "Length");
    lua_pushinteger(L, numElements);
    lua_settable(L, -3);
    // read Elements and push
    for(index=0; index<numElements; index++)
    {
        const mxArray* element = mxGetCell(src, index);
		// revisied: index starts from 1
        lua_pushinteger(L, (index+1));
        if(element == NULL)
            lua_pushstring(L, "NULL");
        else
            readAndPushMxArray(L, element);
        lua_settable(L, -3);
    }    
}


void readAndPushMxArray(lua_State *L, const mxArray* src){
     // get dimensions
    mwSize ndims = mxGetNumberOfDimensions(src);
    const mwSize *dims = mxGetDimensions(src);

    // infer size and stride
    int k;
    THLongStorage *size = THLongStorage_newWithSize(ndims);
    THLongStorage *stride = THLongStorage_newWithSize(ndims);
    for (k=0; k<ndims; k++) {
      THLongStorage_set(size, ndims-k-1, dims[k]);
      if (k > 0)
        THLongStorage_set(stride, ndims-k-1, dims[k-1]*THLongStorage_get(stride,ndims-k));
      else
        THLongStorage_set(stride, ndims-k-1, 1);
    }
     // depending on type, create equivalent Lua/torch data structure
    if (mxGetClassID(src) == mxDOUBLE_CLASS) {
      THDoubleTensor *tensor = THDoubleTensor_newWithSize(size, stride);
      memcpy((void *)(THDoubleTensor_data(tensor)),
             (void *)(mxGetPr(src)), THDoubleTensor_nElement(tensor) * sizeof(double));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.DoubleTensor"));   

    } else if (mxGetClassID(src) == mxSINGLE_CLASS) {
      THFloatTensor *tensor = THFloatTensor_newWithSize(size, stride);
      memcpy((void *)(THFloatTensor_data(tensor)),
             (void *)(mxGetPr(src)), THFloatTensor_nElement(tensor) * sizeof(float));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.FloatTensor"));

    } else if (mxGetClassID(src) == mxINT32_CLASS) {
      THIntTensor *tensor = THIntTensor_newWithSize(size, stride);
      memcpy((void *)(THIntTensor_data(tensor)),
             (void *)(mxGetPr(src)), THIntTensor_nElement(tensor) * sizeof(int));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.IntTensor"));

    } else if (mxGetClassID(src) == mxUINT32_CLASS) {
      THIntTensor *tensor = THIntTensor_newWithSize(size, stride);
      memcpy((void *)(THIntTensor_data(tensor)),
             (void *)(mxGetPr(src)), THIntTensor_nElement(tensor) * sizeof(int));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.IntTensor"));
    } else if ((mxGetClassID(src) == mxINT16_CLASS)) {
      THShortTensor *tensor = THShortTensor_newWithSize(size, stride);
      memcpy((void *)(THShortTensor_data(tensor)),
             (void *)(mxGetPr(src)), THShortTensor_nElement(tensor) * sizeof(short));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ShortTensor"));

    } else if ((mxGetClassID(src) == mxUINT16_CLASS)) {
      THShortTensor *tensor = THShortTensor_newWithSize(size, stride);
      memcpy((void *)(THShortTensor_data(tensor)),
             (void *)(mxGetPr(src)), THShortTensor_nElement(tensor) * sizeof(short));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ShortTensor"));
    } else if (mxGetClassID(src) == mxINT8_CLASS) {
      THCharTensor *tensor = THCharTensor_newWithSize(size, stride);
      memcpy((void *)(THCharTensor_data(tensor)),
             (void *)(mxGetPr(src)), THCharTensor_nElement(tensor) * sizeof(char));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.CharTensor"));
	} else if (mxGetClassID(src) == mxCHAR_CLASS) {
      mwSize numElements = mxGetNumberOfElements(src);
      char* tmpStr = (char*)calloc(numElements+1, sizeof(char));
      mxGetString(src, tmpStr, (numElements+1) * sizeof(char));
	  lua_pushstring(L, tmpStr);
	  free(tmpStr);
    } else if ((mxGetClassID(src) == mxUINT8_CLASS)) {
      THByteTensor *tensor = THByteTensor_newWithSize(size, stride);
      memcpy((void *)(THByteTensor_data(tensor)),
             (void *)(mxGetPr(src)), THByteTensor_nElement(tensor) * sizeof(char));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ByteTensor"));
    } else if ((mxGetClassID(src) == mxLOGICAL_CLASS)) {
      THByteTensor *tensor = THByteTensor_newWithSize(size, stride);
      memcpy((void *)(THByteTensor_data(tensor)),
             (void *)(mxGetPr(src)), THByteTensor_nElement(tensor) * sizeof(char));
      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ByteTensor"));
    }else {
      if ((mxGetClassID(src) == mxCELL_CLASS)) {
        pushMxCellData(L, src, ndims, dims);
      } else if ((mxGetClassID(src) == mxSTRUCT_CLASS)) {
        pushMxStructData(L, src, ndims, dims); 
      } else if ((mxGetClassID(src) == mxINT64_CLASS)) {
        lua_pushstring(L, "unsupported type: mxINT64_CLASS");
      } else if ((mxGetClassID(src) == mxUINT64_CLASS)) {
        lua_pushstring(L, "unsupported type: mxUINT64_CLASS");
      } else if ((mxGetClassID(src) == mxFUNCTION_CLASS)) {
        lua_pushstring(L, "unsupported type: mxFUNCTION_CLASS");
      } else {
        lua_pushstring(L, "unknown type");
      }
    }
    
}

// Loader
static int load_l(lua_State *L) {
  // get args
  const char *path = lua_tostring(L,1);

  // open file
  MATFile *file = matOpen(path, "r");
  if (file == NULL) THError("Error opening file %s", file);

  // create table to hold loaded variables
  lua_newtable(L);  // vars = {}
  int vars = lua_gettop(L);
  int varidx = 1;

  // extract each var
  while (true) {
    // get var+name
    const char *name;
    mxArray *pa = matGetNextVariable(file, &name);
    if (pa == NULL) break;

    lua_pushstring(L, name);    // push varName
    readAndPushMxArray(L, pa);    // push Data
    lua_rawset(L, vars);        // Pop    [key - value] pair
  
    mxDestroyArray(pa);
  }

  // cleanup
  matClose(file);

  // return table 'vars'
  return 1;
}

static mxArray* assignData(lua_State *L, int pos)
{
	mxArray *out = NULL;
	switch(lua_type(L, -1))
	{
		case LUA_TNIL:			
		{
			int32_t val = (int32_t)lua_tointeger(L, pos);
			mwSize size[] = {1,-1,-1,-1,-1,-1,-1,-1};
			const long ndims = 1;			
			out = mxCreateNumericArray(ndims, size, mxINT32_CLASS, mxREAL);
			memcpy((void *)(mxGetPr(out)),  (void *)(&val), sizeof(int32_t));
		}	
			break;
		case LUA_TBOOLEAN:
		{
			int8_t val = (int8_t)lua_toboolean(L,pos);
			mwSize size[] = {1,-1,-1,-1,-1,-1,-1,-1};
			const long ndims = 1;			
			out = mxCreateNumericArray(ndims, size, mxINT8_CLASS, mxREAL);
			memcpy((void *)(mxGetPr(out)),  (void *)(&val), sizeof(int8_t));			
			break;
		}
		case LUA_TNUMBER:
		{
			double val = (double)lua_tonumber(L, pos);
			mwSize size[] = {1,-1,-1,-1,-1,-1,-1,-1};
			const long ndims = 1;			
			out = mxCreateNumericArray(ndims, size, mxDOUBLE_CLASS, mxREAL);
			memcpy((void *)(mxGetPr(out)),  (void *)(&val), sizeof(double));
			break;
		}
		case LUA_TSTRING:
		{
			const char* val = (const char*)lua_tostring(L, pos);			
			out = mxCreateString(val);
			break;
		}
		case LUA_TTABLE:
		{
			mwSize dims[2] = {1, 1};			
			T_ARRAY* array = constractT_ARRAY();			
// 			lua_pushvalue(L, pos);
			lua_pushnil(L);  // first key			
			out = mxCreateStructArray(2, dims, 0, NULL);
			while (lua_next(L, -2) != 0) {
				const char* name = NULL;
				mxArray* data = NULL;
				uint32_t name_idx = 0;
				char tmpStr[100] = {0};
				if(lua_type(L, -2) == LUA_TNUMBER)
				{
					name_idx = (uint32_t)lua_tonumber(L, -2);
					sprintf(tmpStr, "%u", name_idx);
					name = tmpStr;
				}else{
					name = lua_tostring(L,-2);
					name_idx = atoi(name);
				}
				
				data = assignData(L, -1);
				if(data == NULL)
				{
					printf("[mattorch.save] WARNING: '%s' is ignored as it is UNSUPPORTED TYPE\n", name);
				}else{					
					if(name[0] >= '0' && name[0] <= '9')
					{
						addT_ARRAY_item(array, name_idx, data);
					}else{
						int fieldNum = mxAddField(out, name);
						if(fieldNum >= 0)
							mxSetFieldByNumber(out, 0, fieldNum, data);	
					}
				}
				lua_pop(L, 1);
			 }
// 			 lua_pop(L, 1);
			// add array elements			
			if(array->numOfItem > 0)
			{
				mwSize nDim[2] = {1, 1};
				mxArray* cell = NULL;
				T_ARRAY_ITEM* pt = array->first;
				int fieldNum = mxAddField(out, "array");
				nDim[1] = array->numOfItem;
				cell = mxCreateCellArray(2, nDim);
				while(pt != NULL)
				{
					mxSetCell(cell, (pt->idx)-1, mxDuplicateArray(pt->value));
					pt = pt->next;
				}
				
				if(fieldNum >= 0)					
					mxSetFieldByNumber(out, 0, fieldNum, cell);	
			}
			
			distoryT_ARRAY(array);

			break;
		}
		case LUA_TUSERDATA: // Torch.Tensor
		if(luaT_isudata(L, pos, "torch.DoubleTensor"))
		{
			THDoubleTensor *tensor = (THDoubleTensor *)luaT_checkudata(L, -1, luaT_checktypename2id(L, "torch.DoubleTensor"));
			THDoubleTensor *tensorc = THDoubleTensor_newContiguous(tensor);

			// infer size and stride
			int k;
			mwSize size[] = {-1,-1,-1,-1,-1,-1,-1,-1};
			const long ndims = tensorc->nDimension;
			for (k=0; k<ndims; k++) {
				size[k] = tensor->size[ndims-k-1];
			}

			// create matlab array
			out = mxCreateNumericArray(ndims, size, mxDOUBLE_CLASS, mxREAL);

			// copy tensor into array
			memcpy((void *)(mxGetPr(out)), 
				(void *)(THDoubleTensor_data(tensorc)),
				THDoubleTensor_nElement(tensor) * sizeof(double));
		
			// done
			THDoubleTensor_free(tensorc);
		}else if(luaT_isudata(L, pos, "torch.FloatTensor"))
		{
			THFloatTensor *tensor = (THFloatTensor *)luaT_checkudata(L, -1, luaT_checktypename2id(L, "torch.FloatTensor"));
			THFloatTensor *tensorc = THFloatTensor_newContiguous(tensor);

			// infer size and stride
			int k;
			mwSize size[] = {-1,-1,-1,-1,-1,-1,-1,-1};
			const long ndims = tensorc->nDimension;
			for (k=0; k<ndims; k++) {
				size[k] = tensor->size[ndims-k-1];
			}

			// create matlab array
			out = mxCreateNumericArray(ndims, size, mxSINGLE_CLASS, mxREAL);

			// copy tensor into array
			memcpy((void *)(mxGetPr(out)), 
				(void *)(THFloatTensor_data(tensorc)),
				THFloatTensor_nElement(tensor) * sizeof(float));

			// done
			THFloatTensor_free(tensorc);
		}
		break;
	}
	return out;
}

// Save single data
static int save_tensor_l(lua_State *L) {
  // open file for output
  const char *path = lua_tostring(L,1);
  MATFile *file = matOpen(path, "w");
  \
// save it, in a dummy var named 'x'
  const char *name = "x";
  
  mxArray* pm = assignData(L, 2);
  matPutVariable(file, name, pm);
  matClose(file);
  mxDestroyArray(pm);
  return 0;
}

// Save table of tensors
static int save_table_l(lua_State *L) {
  // open file for output
  const char *path = lua_tostring(L,1);
  MATFile *file = matOpen(path, "w");

  mxArray **pms;
  pms = (mxArray**) malloc(sizeof(mxArray*)*1024);
  int counter = 0;
  
  T_ARRAY* array = constractT_ARRAY();
  
  // table is in the stack at index 2 (2nd var)
  lua_pushnil(L);  // first key
  while (lua_next(L, 2) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    const char *name = lua_tostring(L,-2);
	mxArray* pm = NULL;	
	uint32_t name_idx = 0;
	char tmpStr[100] = {0};
	if(lua_type(L, -2) == LUA_TNUMBER)
	{
		name_idx = (uint32_t)lua_tonumber(L, -2);
		sprintf(tmpStr, "%u", name_idx);
		name = tmpStr;
	}else{
		name = lua_tostring(L,-2);
		name_idx = atoi(name);
	}
// 	uint32_t name_idx = atoi(name);
// 	printf("+ %s: <%d>(%s) (%s)\n", name, lua_type(L, -1), lua_typename(L, -1),  luaT_typename(L, -1));
	
	pm = assignData(L, -1);	
	
	if(pm != NULL)
	{
		if(name[0] >= '0' && name[0] <= '9')
		{
			addT_ARRAY_item(array, name_idx, pm);
		}else{
			pms[counter++] = pm;
			matPutVariable(file, name, pm);
		}
	}else
		printf("[mattorch.save] WARNING: '%s' is ignored as it is UNSUPPORTED TYPE\n", name);
    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);
  }
  // add array elements
  if(array->numOfItem > 0)
  {
	  mwSize nDim[2] = {1, 1};
	  mxArray* cell = NULL;
	  T_ARRAY_ITEM* pt = array->first;
	  nDim[1] = array->numOfItem;
	  cell = mxCreateCellArray(2, nDim);
	  while(pt != NULL)
	  {
		  mxSetCell(cell, (pt->idx)-1, mxDuplicateArray(pt->value));
		  pt = pt->next;
	  }
	  matPutVariable(file, "array", cell);
  }
  distoryT_ARRAY(array);
  int i = 0;
  for(i=0; i<counter;i++)
    mxDestroyArray(pms[i]);
  
  free(pms);

  // cleanup
  lua_pop(L, 1);
  matClose(file);
  return 0;
}

static int save_tensor_ascii_l(lua_State *L)
{
  // get file descriptor
  THFile *file = luaT_checkudata(L, 1, luaT_checktypename2id(L, "torch.File"));

  // load tensor
  THDoubleTensor *tensor = (THDoubleTensor *)luaT_checkudata(L, 2, luaT_checktypename2id(L, "torch.DoubleTensor"));
  THDoubleTensor *tensorc = THDoubleTensor_newContiguous(tensor);
  double *tensor_data = THDoubleTensor_data(tensorc);

  // get sizes
  const long ndims = tensorc->nDimension;
  if (ndims > 2) {
    THError("matlab ascii only supports 1d or 2d tensors");
  }

  // write all 
  int i;
  if (ndims == 2) {
    for (i = 0; i < tensorc->size[0]; i ++) {
      THFile_writeDoubleRaw(file, tensor_data, tensorc->size[1]);
      tensor_data += tensorc->size[1];
    }
  } else {
    for (i = 0; i < tensorc->size[0]; i ++) {
      THFile_writeDoubleRaw(file, tensor_data, 1);
      tensor_data += 1;
    }
  }

  // cleanup
  THDoubleTensor_free(tensorc);
  return 0;
}

// Register functions in LUA
static const struct luaL_reg matlab [] = {
  {"load", load_l},
  {"saveTensor", save_tensor_l},
  {"saveTable", save_table_l},
  {"saveTensorAscii", save_tensor_ascii_l},
  {NULL, NULL}  /* sentinel */
};

int luaopen_libmattorch (lua_State *L) {
  luaL_openlib(L, "libmattorch", matlab, 0);
  return 1;
}
