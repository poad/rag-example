import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as nodejs from 'aws-cdk-lib/aws-lambda-nodejs';
import * as awslogs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as deployment from 'aws-cdk-lib/aws-s3-deployment';
import { buildFrontend } from './process/setup';
import assert from 'node:assert';

export interface Config extends cdk.StackProps {
  bucketName: string;
  appName: string;
  cloudfront: {
    comment: string;
  };
}

interface CloudfrontCdnTemplateStackProps extends Config {
  environment?: string;
  endpoint: string;
  apiKey: string;
  deployName: string;
  embeddingsDeployName: string;
  apiVersion: string;
  langfuse?: {
    sk: string;
    pk: string;
    endpoint: string;
  };
  pinecone: {
    index: string;
    apiKey?: string;
  },
  langsmith?: {
    apiKey: string;
    project: string;
    endpoint: string;
  };
}

export class CloudfrontCdnTemplateStack extends cdk.Stack {
  constructor(
    scope: Construct,
    id: string,
    props: CloudfrontCdnTemplateStackProps,
  ) {
    super(scope, id, props);

    const {
      bucketName,
      appName,
      environment,
      cloudfront: { comment },
      endpoint,
      apiKey,
      deployName,
      embeddingsDeployName,
      apiVersion,
      langfuse,
      langsmith,
      pinecone,
    } = props;

    assert(pinecone.apiKey);

    buildFrontend();

    const functionName = `${environment ? `${environment}-` : ''}${appName}-api`;
    new awslogs.LogGroup(this, 'ApolloLambdaFunctionLogGroup', {
      logGroupName: `/aws/lambda/${functionName}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      retention: awslogs.RetentionDays.ONE_DAY,
    });

    const devOptions = {
      // environment: {
      //   NODE_OPTIONS: '--enable-source-maps',
      // },
      // bundling: {
      //   sourceMap: true,
      //   sourceMapMode: nodejs.SourceMapMode.BOTH,
      //   sourcesContent: true,
      //   keepNames: true,
      // },
      applicationLogLevelV2: lambda.ApplicationLogLevel.TRACE,
    };

    const apiRootPath = '/api/';

    const langfuseEnv = langfuse ? {
      LANGFUSE_SECRET_KEY: langfuse.sk,
      LANGFUSE_PUBLIC_KEY: langfuse.pk,
      ...(langfuse.endpoint ? {
        LANGFUSE_BASEURL: langfuse.endpoint,
      } : {}),
    } : {};

    const langsmithEnv: Record<string, string> = langsmith ? {
      LANGCHAIN_TRACING_V2: 'true',
      LANGCHAIN_ENDPOINT: langsmith.endpoint,
      LANGCHAIN_API_KEY: langsmith.apiKey,
      LANGCHAIN_PROJECT: langsmith.project,
    } : {};

    const fn = new nodejs.NodejsFunction(this, 'Lambda', {
      runtime: lambda.Runtime.NODEJS_22_X,
      architecture: lambda.Architecture.ARM_64,
      entry: './lambda/index.ts',
      functionName,
      retryAttempts: 0,
      environment: {
        // ...devOptions.environment,
        API_ROOT_PATH: apiRootPath,
        AZURE_OPENAI_API_INSTANCE_NAME: endpoint,
        AZURE_OPENAI_API_DEPLOYMENT_NAME: deployName,
        AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME: embeddingsDeployName,
        AZURE_OPENAI_API_KEY: apiKey,
        AZURE_OPENAI_API_VERSION: apiVersion,
        PINECONE_API_KEY: pinecone.apiKey,
        PINECONE_INDEX: pinecone.index,
        ...langfuseEnv,
        ...langsmithEnv,
      },
      bundling: {
        minify: true,
        // ...devOptions.bundling,
      },
      memorySize: 256,
      timeout: cdk.Duration.minutes(1),
      role: new iam.Role(this, 'ApolloLambdaFunctionExecutionRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLambdaExecute'),
          iam.ManagedPolicy.fromAwsManagedPolicyName('CloudFrontReadOnlyAccess'),
        ],
        inlinePolicies: {
          'bedrock-policy': new iam.PolicyDocument({
            statements: [
              new iam.PolicyStatement({
                effect: iam.Effect.ALLOW,
                actions: [
                  'bedrock:InvokeModel*',
                  'logs:PutLogEvents',
                ],
                resources: ['*'],
              }),
            ],
          }),
        },
      }),
      loggingFormat: lambda.LoggingFormat.JSON,
      applicationLogLevelV2: devOptions.applicationLogLevelV2,
    });

    const s3bucket = new s3.Bucket(this, 'S3Bucket', {
      bucketName,
      versioned: false,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    const websiteIndexPageForwardFunction = new cloudfront.Function(this, 'WebsiteIndexPageForwardFunction', {
      functionName: `${appName}-api-index-forword`,
      code: cloudfront.FunctionCode.fromFile({
        filePath: 'function/index.js',
      }),
      runtime: cloudfront.FunctionRuntime.JS_2_0,
    });
    const functionAssociations = [
      {
        eventType: cloudfront.FunctionEventType.VIEWER_REQUEST,
        function: websiteIndexPageForwardFunction,
      },
    ];
    const originAccessControl = new cloudfront.S3OriginAccessControl(this, 'S3OAC', {
      originAccessControlName: `OAC for S3 (${appName}-api)`,
      signing: cloudfront.Signing.SIGV4_NO_OVERRIDE,
    });

    const cf = new cloudfront.Distribution(this, 'CloudFront', {
      comment,
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3bucket, {
          originAccessControl,
          originAccessLevels: [cloudfront.AccessLevel.READ],
          originId: 's3',
        }),
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        functionAssociations,
      },
      additionalBehaviors: {
        [`${apiRootPath}*`]: {
          origin: new origins.FunctionUrlOrigin(fn.addFunctionUrl({
            authType: cdk.aws_lambda.FunctionUrlAuthType.AWS_IAM,
            invokeMode: cdk.aws_lambda.InvokeMode.RESPONSE_STREAM,
          }),
          {
            originId: 'lambda',
            readTimeout: cdk.Duration.minutes(1),
          }),
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
          responseHeadersPolicy: new cdk.aws_cloudfront.ResponseHeadersPolicy(
            this,
            'ResponseHeadersPolicy',
            {
              corsBehavior: {
                accessControlAllowOrigins: [
                  'http://localhost:4173',
                  'http://localhost:5173',
                ],
                accessControlAllowHeaders: ['*'],
                accessControlAllowMethods: ['ALL'],
                accessControlAllowCredentials: false,
                originOverride: true,
              },
            },
          ),
        },
      },
      httpVersion: cloudfront.HttpVersion.HTTP2_AND_3,
    });

    const deployRole = new iam.Role(this, 'DeployWebsiteRole', {
      roleName: `${appName}-deploy-role`,
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      inlinePolicies: {
        's3-policy': new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ['s3:*'],
              resources: [`${s3bucket.bucketArn}/`, `${s3bucket.bucketArn}/*`],
            }),
          ],
        }),
      },
    });

    new deployment.BucketDeployment(this, 'DeployWebsite', {
      sources: [deployment.Source.asset(`${process.cwd()}/../app/dist`)],
      destinationBucket: s3bucket,
      destinationKeyPrefix: '/',
      exclude: ['.DS_Store', '*/.DS_Store'],
      prune: true,
      retainOnDelete: false,
      role: deployRole,
    });

    // OAC for Lambda
    const cfnOriginAccessControl =
      new cdk.aws_cloudfront.CfnOriginAccessControl(
        this,
        'OriginAccessControl',
        {
          originAccessControlConfig: {
            name: `OAC for Lambda Functions URL (${functionName})`,
            originAccessControlOriginType: 'lambda',
            signingBehavior: 'always',
            signingProtocol: 'sigv4',
          },
        },
      );

    const cfnDistribution = cf.node.defaultChild as cdk.aws_cloudfront.CfnDistribution;

    // Set OAC for Lambda
    cfnDistribution.addPropertyOverride(
      'DistributionConfig.Origins.1.OriginAccessControlId',
      cfnOriginAccessControl.attrId,
    );

    // Add permission Lambda Function URLs
    fn.addPermission('AllowCloudFrontServicePrincipal', {
      principal: new iam.ServicePrincipal('cloudfront.amazonaws.com'),
      action: 'lambda:InvokeFunctionUrl',
      sourceArn: `arn:aws:cloudfront::${cdk.Stack.of(this).account}:distribution/${cf.distributionId}`,
    });

    new cdk.CfnOutput(this, 'AccessURLOutput', {
      value: `https://${cf.distributionDomainName}`,
    });
  }
}
