

后端提供的接口：

Swagger接口地址：http://192.168.9.39/edu-api/avatar/v3/api-docs

1、获取RAG知识库列表

/api/digitalcall/getDatasetList?teamId=&id=&firstId=&size=

参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "name": "知识库1",

​      "isEnable": 1,

​      "fileCount": 0,

​      "updateTime": "2025-12-17 11:10:47"

​    },

​    {

​      "id": 2,

​      "teamId": 40,

​      "name": "知识库2",

​      "isEnable": 1,

​      "fileCount": 0,

​      "updateTime": "2025-12-17 10:58:53"

​    }

  ],

  "success": **true**

}

返回字段：

Id：     主键id

teamId：   团队id

ragDatasetId：RAG系统知识库ID

name：   知识库名称

isEnable：  是否生效：0=不生效，1=生效

fileCount：  文件数量

updateTime： 更新时间



2、获取RAG文档列表

/api/digitalcall/getDatasetDocumentList?teamId=&id=&firstId=&size=

参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "datasetId": 1,

​      "ragDocumentId": "123",

​      "resourceId": 48256,

​      "documentName": "知识文档1",

​      "startTime": "2025-12-17 14:23:30",

​      "endTime": "2025-12-31 14:23:35",

​      "isEnable": 1,

​      "isEnhance": 1,

​      "status": 0,

​      "segmentFlag": 0,

​      "fileSize": 18487,

​      "updateTime": "2025-12-17 14:38:28"

​    },

​    {

​      "id": 2,

​      "teamId": 40,

​      "datasetId": 1,

​      "ragDocumentId": "123",

​      "resourceId": 48257,

​      "documentName": "知识文档2",

​      "startTime": "2025-12-17 14:23:33",

​      "endTime": "2025-12-31 14:23:38",

​      "isEnable": 1,

​      "isEnhance": 1,

​      "status": 0,

​      "segmentFlag": 0,

​      "fileSize": 16743097,

​      "updateTime": "2025-12-17 14:38:28"

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

datasetId：    知识库id

ragDocumentId： RAG系统文档ID

resourceId：   系统资源ID

documentName：文档名称

startTime：   生效开始时间

endTime：   生效结束时间

isEnable：   是否生效：0=不生效，1=生效

isEnhance：  是否知识增强：0=不增强，1=增强

status：    状态：0=未学习，1=学习中，2=学习成功，3=学习失败，4=知识增强中

segmentFlag： 分段策略：0=自动分段与数据清洗，1=自定义文档分段与数据清洗策略

updateTime： 更新时间



3、获取FAQ问答列表

/api/digitalcall/getDatasetFaqList?teamId=&id=&firstId=&size=



http://192.168.9.39/edu-api/avatar/api/digitalcall/getDatasetFaqList?teamId=&id=&firstId=&size=



参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 8,

​      "teamId": 40,

​      "questionName": "问题101",

​      "startTime": "2025-12-16 18:09:45",

​      "endTime": "2099-12-31 23:23:59",

​      "isEnable": 1,

​      "isClear": 1,

​      "similarQuestions": [

​        {

​          "id": 11,

​          "name": "相似问题101"

​        },

​        {

​          "id": 12,

​          "name": "相似问题102"

​        },

​        {

​          "id": 13,

​          "name": "相似问题103"

​        }

​      ],

​      "answers": [

​        {

​          "id": 14,

​          "name": "答案101"

​        },

​        {

​          "id": 15,

​          "name": "答案102"

​        }

​      ]

​    },

​    {

​      "id": 9,

​      "teamId": 40,

​      "questionName": "问题202",

​      "startTime": "2025-12-16 18:45:02",

​      "endTime": "2099-12-31 23:23:59",

​      "isEnable": 1,

​      "isClear": 1,

​      "similarQuestions": [

​        {

​          "id": 10,

​          "name": "相似问题201"

​        }

​      ],

​      "answers": [

​        {

​          "id": 13,

​          "name": "答案201"

​        }

​      ]

​    }

  ],

  "success": **true**

}

返回字段：

id：      主键id

teamId：    团队id

questionName： 标准问题

startTime：   生效开始时间

endTime：   生效结束时间

isEnable：   是否生效：0=不生效，1=生效

isClear：    是否澄清：0=不澄清，1=澄清

updateTime：  更新时间

similarQuestions

id：    相似问题id

name：  相似问题

Answers

id：    通用答案id

name：  通用答案





4、获取视频资源列表

/api/digitalcall/getDatasetVideoList?teamId=&id=&firstId=&size=

参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "ragVideoId": "123",

​      "resourceId": 48256,

​      "videoName": "视频测试1",

​      "startTime": "2025-12-19 10:16:30",

​      "endTime": "2025-12-31 10:16:33",

​      "isEnable": 1,

​      "isEnhance": 1,

​      "status": 0,

​      "fileSize": 18487

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

ragVideoId：   RAG视频ID

resourceId：   系统资源ID

VideoName：  视频资源名称

startTime：   生效开始时间

endTime：   生效结束时间

isEnable：   是否生效：0=不生效，1=生效

isEnhance：  是否知识增强：0=不增强，1=增强

status：    状态：0=未学习，1=学习中，2=学习成功，3=学习失败，4=知识增强中

fileSize：   文件大小，单位字节

updateTime： 更新时间



5、获取资源的下载地址

/api/digitalcall/getResourceUrl?resourceId=123

参数：

resourceId：资源id，必填

返回内容（文件的rul地址）：

https://education-test-private.oss-cn-beijing.aliyuncs.com/application/8947790a-0a41-4c99-9d0c-d1f943f6a84b.docx?Expires=1766123486&OSSAccessKeyId=LTAI4FctZ3DLxBxVPrqD4sCo&Signature=s0gyUAnU0zYsG4IMstaHpIWqBqQ%3D



6、获取专业词库列表

/api/digitalcall/getThesaurusMajorList?teamId=&id=&firstId=&size=

参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "thesaurusName": "专业词库一",

​      "wordCount": 10,

​      "lastUserName": "张三",

​      "updateTime": "2025-12-31 10:16:33",

​      "words": [

​        {

​          "id": 1,

​          "teamId": 40,

​          "thesaurusId": 1,

​          "wordName": "专业词条名称201"，

​          "similarWordName": "同义词条名称401，同义词条名称402"

​        }

​      ]

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

thesaurusName： 专业词库名称

wordCount：   词条数量

lastUserName：  最后操作人名称

updateTime：   更新时间

words：     专业词条列表

id：       词条主键id

teamId：     团队id

thesaurusId：   专业词库id

wordName：   专业词条名称

similarWordName： 同义词条名称

 

7、获取专业词条列表

/api/digitalcall/getThesaurusMajorWordList?thesaurusId=teamId=&id=&firstId=&size=

参数：

thesaurusId：专业词库id，非必填

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "thesaurusId": 1,

​      "wordName": "专业词条名称201"，

​      "similarWordName": "同义词条名称401，同义词条名称402"

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

thesaurusId：   专业词库id

wordName：   专业词条名称

similarWordName： 同义词条名称

 

8、获取敏感词库列表

/api/digitalcall/getThesaurusSensitiveList?teamId=&id=&firstId=&size=

参数：

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "thesaurusName": "敏感词库一",

​      "wordCount": 10,

​      "lastUserName": "张三",

​      "updateTime": "2025-12-31 10:16:33",

​      "words": [

​        {

​          "id": 1,

​          "teamId": 40,

​          "thesaurusId": 1,

​          "wordName": "敏感词条名称201"        }

​      ]

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

thesaurusName： 敏感词库名称

wordCount：   词条数量

lastUserName：  最后操作人名称

updateTime：   更新时间

words：     敏感词条列表

id：       词条主键id

teamId：     团队id

thesaurusId：   敏感词库id

wordName：   敏感词条名称

 

9、获取敏感词条列表

/api/digitalcall/getThesaurusSensitiveWordList?thesaurusId=teamId=&id=&firstId=&size=

参数：

thesaurusId：敏感词库id，非必填

teamId：团队id，非必填

Id：  主键id，非必填

firstId： 获取id大于此firstId值的记录，非必填

size：  每次获取记录的条数，非必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "data": [

​    {

​      "id": 1,

​      "teamId": 40,

​      "thesaurusId": 1,

​      "wordName": "敏感词条名称201"

​    }

  ],

  "success": **true**

}

返回字段：

id：       主键id

teamId：     团队id

thesaurusId：   敏感词库id

wordName：   敏感词条名称

 

10、保存用户对话记录

/api/digitalcall/saveUserDialog

参数JSON格式：

{

  "teamId": 40,

  "employeeId": 1,

  "deviceId": "CJQX-YJO1",

  "deviceName": "数字人全息舱 DSee型号",

  "dialogs": [

​    {

​      "userType": 0,

​      "userId": 1,

​      "userName": "小欧",

​      "headUrl": "",

​      "dialogContent": "您好，能为你做些什么",

​      "dialogTime": "2025-12-22 16:22:26",

​      "isLike": 0

​    },

​    {

​      "userType": 1,

​      "userId": 10197,

​      "userName": "团队创建人",

​      "headUrl": "",

​      "dialogContent": "今天星期几",

​      "dialogTime": "2025-12-22 16:22:34",

​      "isLike": 0

​    },

​    {

​      "userType": 0,

​      "userId": 1,

​      "userName": "小欧",

​      "headUrl": "",

​      "dialogContent": "今天星期一",

​      "dialogTime": "2025-12-22 16:22:39",

​      "isLike": 0

​    },

​    {

​      "userType": 1,

​      "userId": 10197,

​      "userName": "团队创建人",

​      "headUrl": "",

​      "dialogContent": "谢谢，再见",

​      "dialogTime": "2025-12-22 16:22:45",

​      "isLike": 0

​    },

​    {

​      "userType": 0,

​      "userId": 1,

​      "userName": "小欧",

​      "headUrl": "",

​      "dialogContent": "再见，很高兴为您服务",

​      "dialogTime": "2025-12-22 16:22:47",

​      "isLike": 0

​    }

  ]

}

参数字段：

teamId：   团队id，必填

employeeId： 数字员工id，必填

deviceId：   设备编号，必填

deviceName： 设备名称，必填

Dialogs：    用户对话详细信息列表

userType：   用户类型：0=数字员工id，1=用户id，必填

userId：    用户id，必填

userName：   用户名称，必填

headUrl：    用户头像，非必填

dialogContent： 对话内容，必填

dialogTime：  对话时间，必填

isLike：     是否点赞：0=默认，1=赞，2=不赞，必填

返回内容：

{

  "status": 200,

  "message": "ok",

  "success": **true**

}

返回字段：

status：   200=保存成功，1=保存失败

Message：  失败原因

 