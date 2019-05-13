/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_DEFINITIONS_
#define CUHD_DEFINITIONS_

#define cuhd_buf(TYPE, IDENTIFIER) std::unique_ptr<TYPE[]> IDENTIFIER
#define cuhd_out_buf(TYPE, IDENTIFIER) std::unique_ptr<TYPE[]> IDENTIFIER

#endif /* CUHD_DEFINITIONS_H_ */
