#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <memory>

#include <VapourSynth/VapourSynth.h>
#include <VapourSynth/VSHelper.h>

struct mxnetData
{
	VSNodeRef *node;
	VSVideoInfo vi;
	int patchWidth, patchHeight;
	float *srcBuffer, *dstBuffer;
	PredictorHandle hPred;
};

class BufferFile
{
public:
	std::string file_path_;
	size_t length_;
	char* buffer_;

	explicit BufferFile(std::string file_path)
		:file_path_(file_path)
	{
		std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
		if (!ifs) {
			std::cerr << "Can't open the file. Please check " << file_path << ". \n";
			length_ = 0;
			buffer_ = NULL;
			return;
		}

		ifs.seekg(0, std::ios::end);
		length_ = ifs.tellg();
		ifs.seekg(0, std::ios::beg);

		buffer_ = new char[sizeof(char) * length_];
		ifs.read(buffer_, length_);
		ifs.close();
	}

	size_t GetLength()
	{
		return length_;
	}
	char* GetBuffer()
	{
		return buffer_;
	}

	~BufferFile()
	{
		if (buffer_) {
			delete[] buffer_;
			buffer_ = NULL;
		}
	}
};

static int process(const VSFrameRef *src, VSFrameRef *dst, mxnetData * VS_RESTRICT d, const VSAPI * vsapi) noexcept
{
	if (d->vi.format->colorFamily != cmYUV || d->vi.format->numPlanes != 3 || d->vi.format->subSamplingH || d->vi.format->subSamplingW) {
		return 3;
	}

	const unsigned width = vsapi->getFrameWidth(src, 0);
	const unsigned height = vsapi->getFrameHeight(src, 0);
	const unsigned srcStride = vsapi->getStride(src, 0) / sizeof(float);
	const float * srcpY = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0));
	const float * srcpU = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 1));
	const float * srcpV = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 2));

	int imageSize = width * height * d->vi.format->numPlanes;

	mx_float* ptr_image_y = d->srcBuffer;
	mx_float* ptr_image_u = d->srcBuffer + imageSize / 3;
	mx_float* ptr_image_v = d->srcBuffer + imageSize / 3 * 2;

	for (unsigned y = 0; y < height; y++) {
		for (unsigned x = 0; x < width; x++) {
			*ptr_image_y++ = srcpY[x];
			*ptr_image_u++ = srcpU[x] + 0.5f;
			*ptr_image_v++ = srcpV[x] + 0.5f;
		}

		srcpY += srcStride;
		srcpU += srcStride;
		srcpV += srcStride;
	}

	if (MXPredSetInput(d->hPred, "data", d->srcBuffer, imageSize) != 0) {
		return 2;
	}
	if (MXPredForward(d->hPred) != 0) {
		return 2;
	}

	mx_uint output_index = 0;

	mx_uint *shape = nullptr;
	mx_uint shape_len = 0;

	// Get Output Result
	if (MXPredGetOutputShape(d->hPred, output_index, &shape, &shape_len) != 0) {
		return 2;
	}

	mx_uint outputSize = 1;
	for (mx_uint i = 0; i < shape_len; ++i) outputSize *= shape[i];

	if (outputSize != imageSize) {
		return 1;
	}

	if (MXPredGetOutput(d->hPred, output_index, d->dstBuffer, outputSize) != 0) {
		return 2;
	}

	for (unsigned i = 0; i < (unsigned)d->vi.format->numPlanes; ++i) {
		const unsigned stride = vsapi->getStride(dst, i) / sizeof(float);
		float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, i));

		for (unsigned y = 0; y < height; y++) {
			std::copy(d->dstBuffer, d->dstBuffer + width * sizeof(float), dstp);
		}

		dstp += stride;
		d->dstBuffer += height;
	}

	return 0;
}

static const VSFrameRef *VS_CC mxdncnnGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
	mxnetData *d = (mxnetData *)* instanceData;

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, d->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
		VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);

		const auto error = process(src, dst, d, vsapi);
		if (error != 0) {
			const char * err = "";

			if (error == 1)
				err = "invalid parameter";
			else if (error == 2)
				err = "failed process mxnet";
			else if (error == 3)
				err = "not support color family";

			vsapi->setFilterError((std::string{ "mxDncnn: " } +err).c_str(), frameCtx);
			vsapi->freeFrame(src);
			vsapi->freeFrame(dst);
			return nullptr;
		}

		vsapi->freeFrame(src);
		return dst;
	}

	return 0;
}

static void VS_CC mxdncnnFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
	mxnetData *d = (mxnetData *)instanceData;
	vsapi->freeNode(d->node);

	MXPredFree(d->hPred);

	delete[] d->srcBuffer;
	delete[] d->dstBuffer;

	free(d);
}

static void VS_CC mxdncnnInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
	mxnetData * d = static_cast<mxnetData *>(*instanceData);
	vsapi->setVideoInfo(&d->vi, 1, node);
}

static void VS_CC mxdncnnCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	int err;

	mxnetData d{};

	d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
	d.vi = *vsapi->getVideoInfo(d.node);

	try {
		if (!isConstantFormat(&d.vi) || d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32)
			throw std::string{ "only constant format 32 bit float input supported" };

		const int param = int64ToIntS(vsapi->propGetInt(in, "param", 0, &err));

		d.patchWidth = int64ToIntS(vsapi->propGetInt(in, "patch_w", 0, &err));
		if (err)
			d.patchWidth = d.vi.width;

		d.patchHeight = int64ToIntS(vsapi->propGetInt(in, "patch_h", 0, &err));
		if (err)
			d.patchHeight = d.vi.height;

		const int ctx = int64ToIntS(vsapi->propGetInt(in, "ctx", 0, &err));

		const int dev_id = int64ToIntS(vsapi->propGetInt(in, "dev_id", 0, &err));

		if (ctx != 1 && ctx != 2 && ctx != 0)
			throw std::string{ "context must be 1(cpu) or 2(gpu)" };

		if (d.patchWidth < 1)
			throw std::string{ "patch_w must be greater than or equal to 1" };

		if (d.patchHeight < 1)
			throw std::string{ "patch_h must be greater than or equal to 1" };

		if (dev_id < 0)
			throw std::string{ "device id must be greater than or equal to 0" };

		if (d.vi.format->colorFamily == cmYUV) {
			d.srcBuffer = new (std::nothrow) float[d.vi.width * d.vi.height * 3];
			d.dstBuffer = new (std::nothrow) float[d.vi.width * d.vi.height * 3];
			if (!d.srcBuffer || !d.dstBuffer)
				throw std::string{ "malloc failure (buffer)" };
		} else {
			throw std::string{ "only support YUV444" };
		}

		const std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginById("com.kice.mxDncnn", core)) };
		std::string dataPath{ pluginPath.substr(0, pluginPath.find_last_of('/')) };

		std::string modelPath = dataPath + "/dncnn/DnCNN-symbol.json";

		auto paramPath = dataPath;
		if (param == 0)
			paramPath += "/dncnn/DnCNN88-0000.params";
		else
			paramPath += "/dncnn/DnCNN" + std::to_string(param) + "-0000.params";

		BufferFile json_data(modelPath);
		BufferFile param_data(paramPath);

		d.hPred = 0;

		// Parameters
		int dev_type = ctx == 0 ? 1 : 2;
		mx_uint num_input_nodes = 1;
		const char* input_key[1] = { "data" };
		const char** input_keys = input_key;

		const mx_uint input_shape_indptr[] = { 0, 4 };
		const mx_uint input_shape_data[4] =
		{
			1,
			static_cast<mx_uint>(3),
			static_cast<mx_uint>(d.patchHeight),
			static_cast<mx_uint>(d.patchWidth)
		};

		d.hPred = 0;

		if (json_data.GetLength() == 0 || param_data.GetLength() == 0)
			throw std::string{ "Cannot open symbol json file or param data file" };

		// Create Predictor
		if (MXPredCreate(
			(const char*)json_data.GetBuffer(),
			(const char*)param_data.GetBuffer(),
			static_cast<int>(param_data.GetLength()),
			dev_type,
			dev_id,
			num_input_nodes,
			input_keys,
			input_shape_indptr,
			input_shape_data,
			&d.hPred) != 0) {
			throw std::string{ "Create Predictor failed" };
		}

		if (d.hPred == 0) {
			throw std::string{ "Invalid Predictor" };
		}
	} catch (const std::string & error) {
		vsapi->setError(out, ("mxDncnn: " + error).c_str());
		vsapi->freeNode(d.node);
		return;
	}

	mxnetData* data = new mxnetData{ d };

	vsapi->createFilter(in, out, "mxDncnn", mxdncnnInit, mxdncnnGetFrame, mxdncnnFree, fmParallelRequests, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
	configFunc("com.kice.mxDncnn", "mx", "A mxnet implement of the paper \"Beyond a Gaussian Denoiser : Residual Learning of Deep CNN for Image Denoising\"", VAPOURSYNTH_API_VERSION, 1, plugin);
	registerFunc("mxDNCNN",
		"clip:clip;"
		"patch_w:int:opt;"
		"patch_h:int:opt;"
		"param:int:opt;"
		"ctx:int:opt;"
		"dev_id:int:opt;",
		mxdncnnCreate, nullptr, plugin);
}
