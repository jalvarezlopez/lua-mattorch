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
        mxCELL_CLASS
        mxSTRUCT_CLASS
        mxLOGICAL_CLASS
        mxCHAR_CLASS      
        mxDOUBLE_CLASS    Y
        mxSINGLE_CLASS    
        mxINT8_CLASS      
        mxUINT8_CLASS     
        mxINT16_CLASS     
        mxUINT16_CLASS    
        mxINT32_CLASS     
        mxUINT32_CLASS    
        mxINT64_CLASS
        mxUINT64_CLASS
        mxFUNCTION_CLASS

  -
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

static void readAndPushMxArray(lua_State *L, const mxArray* src);
static void pushMxCellData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims);
static void pushMxStructData(lua_State *L, const mxArray* src, mwSize ndims, const mwSize *dims);

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

// Save single tensor
static int save_tensor_l(lua_State *L) {
  // open file for output
  const char *path = lua_tostring(L,1);
  MATFile *file = matOpen(path, "w");

  // load tensor
  THDoubleTensor *tensor = (THDoubleTensor *)luaT_checkudata(L, 2, luaT_checktypename2id(L, "torch.DoubleTensor"));
  THDoubleTensor *tensorc = THDoubleTensor_newContiguous(tensor);

  // infer size and stride
  int k;
  mwSize size[] = {-1,-1,-1,-1,-1,-1,-1,-1};
  const long ndims = tensorc->nDimension;
  for (k=0; k<ndims; k++) {
    size[k] = tensor->size[ndims-k-1];
  }

  // create matlab array
  mxArray *pm = mxCreateNumericArray(ndims, size, mxDOUBLE_CLASS, mxREAL);

  // copy tensor
  memcpy((void *)(mxGetPr(pm)), 
         (void *)(THDoubleTensor_data(tensor)),
         THDoubleTensor_nElement(tensor) * sizeof(double));

  // save it, in a dummy var named 'x'
  const char *name = "x";
  matPutVariable(file, name, pm);

  // done
  THDoubleTensor_free(tensorc);
  matClose(file);
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
  // table is in the stack at index 2 (2nd var)
  lua_pushnil(L);  // first key
  while (lua_next(L, 2) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    const char *name = lua_tostring(L,-2);
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
    mxArray *pm = mxCreateNumericArray(ndims, size, mxDOUBLE_CLASS, mxREAL);
    pms[counter++] = pm;

    // copy tensor into array
    memcpy((void *)(mxGetPr(pm)), 
           (void *)(THDoubleTensor_data(tensorc)),
           THDoubleTensor_nElement(tensor) * sizeof(double));

    // store it
    matPutVariable(file, name, pm);

    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);

    // cleanup
    THDoubleTensor_free(tensorc);
  }
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
